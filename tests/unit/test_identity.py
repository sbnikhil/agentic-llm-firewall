"""Identity binding: verified LiteLLM user/team -> policy, server-side only.

Design: docs/design/policy.md section 3. The LiteLLM auth object is faked
here; the adapter task re-checks the field names against the pinned LiteLLM.
"""

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from llmfirewall.engine.decision import Action
from llmfirewall.engine.identity import (
    IdentityMap,
    PolicyResolver,
    VerifiedIdentity,
    load_identity_map,
    parse_identity_map,
)
from llmfirewall.engine.policy import (
    FailureMode,
    Policy,
    PolicyError,
    Role,
    load_policy_file,
    merge_stricter,
    parse_policy,
)
from llmfirewall.engine.settings import REPO_ROOT
from tests.unit.engine_fakes import (
    CHEAP,
    EXPENSIVE,
    FakeDetector,
    ThresholdParams,
    detector_config,
    policy_data,
)

DETECTORS = [
    FakeDetector("rules", CHEAP),
    FakeDetector("scored", EXPENSIVE, params_model=ThresholdParams),
]


def policy(name: str, rules: Any, scored: Any, **options: Any) -> Policy:
    return parse_policy(
        policy_data({"rules": rules, "scored": scored}, **options),
        name=name,
        detectors=DETECTORS,
    )


POLICIES = {
    "default": policy(
        "default",
        detector_config("block", ["user"]),
        detector_config("redact", ["response"], params={"min_score": 0.5}),
    ),
    "strict": policy(
        "strict",
        detector_config("block", ["user", "tool"]),
        detector_config("block", ["user", "response"], params={"min_score": 0.2}),
        latency_budget_ms=500,
    ),
    # Weaker than strict in every field.
    "permissive": policy(
        "permissive",
        detector_config("allow", [], enabled=False),
        detector_config("allow", ["response"], params={"min_score": 0.9}),
        on_detector_error="fail_open",
        on_timeout="fail_open",
        latency_budget_ms=5000,
    ),
    # Stricter than strict in some fields, weaker in others.
    "mixed": policy(
        "mixed",
        detector_config("redact", ["system"]),
        detector_config("allow", ["tool"], params={"min_score": 0.1}),
        on_detector_error="fail_open",
        latency_budget_ms=200,
    ),
    # Every detector disabled, but with the strictest settings: if a disabled
    # side counted in a merge, these would leak into it.
    "dormant": policy(
        "dormant",
        detector_config(
            "block", ["system", "developer", "user", "tool"], enabled=False
        ),
        detector_config(
            "block",
            ["system", "user", "assistant", "tool", "response"],
            enabled=False,
            params={"min_score": 0.0},
        ),
        latency_budget_ms=5000,
    ),
}

MAP_DATA: dict[str, Any] = {
    "version": 1,
    "default": "default",
    "users": {
        "alice": "permissive",
        "carol": "strict",
        "dave": "mixed",
        "erin": "dormant",
    },
    "teams": {
        "contractors": "strict",
        "admins": "permissive",
        "ops": "mixed",
        "idle": "dormant",
    },
}


@pytest.fixture
def resolver() -> PolicyResolver:
    return PolicyResolver(parse_identity_map(MAP_DATA, POLICIES), POLICIES)


def resolve(resolver: PolicyResolver, user_id: Any, team_id: Any) -> str:
    return resolver.resolve(VerifiedIdentity(user_id=user_id, team_id=team_id)).name


# Resolution order ---------------------------------------------------------


def test_unmapped_user_gets_the_default_policy(resolver: PolicyResolver) -> None:
    assert resolve(resolver, "nobody", None) == "default"


def test_unmapped_user_and_team_get_the_default_policy(
    resolver: PolicyResolver,
) -> None:
    assert resolve(resolver, "nobody", "no-such-team") == "default"


def test_caller_without_any_identity_gets_the_default_policy(
    resolver: PolicyResolver,
) -> None:
    assert resolve(resolver, None, None) == "default"


def test_empty_ids_count_as_no_id(resolver: PolicyResolver) -> None:
    assert resolve(resolver, "", "") == "default"


def test_mapped_user_gets_their_policy(resolver: PolicyResolver) -> None:
    assert resolve(resolver, "carol", None) == "strict"


def test_mapped_team_gets_its_policy(resolver: PolicyResolver) -> None:
    assert resolve(resolver, "nobody", "contractors") == "strict"


def test_user_mapped_and_team_unmapped_gets_the_user_policy(
    resolver: PolicyResolver,
) -> None:
    assert resolve(resolver, "alice", "no-such-team") == "permissive"


def test_both_mapped_to_the_same_policy_gets_that_policy(
    resolver: PolicyResolver,
) -> None:
    assert resolve(resolver, "carol", "contractors") == "strict"


# Both mapped: the stricter merge --------------------------------------------


def assert_at_least_as_strict(merged: Policy, other: Policy) -> None:
    closed = FailureMode.FAIL_CLOSED
    if other.on_detector_error is closed:
        assert merged.on_detector_error is closed
    if other.on_timeout is closed:
        assert merged.on_timeout is closed
    assert merged.latency_budget_ms <= other.latency_budget_ms
    assert set(merged.detectors) == set(other.detectors)
    for name, theirs in other.detectors.items():
        # Only effective behavior counts: a disabled detector does nothing,
        # whatever its action, roles or params say.
        if not theirs.enabled:
            continue
        ours = merged.detectors[name]
        assert ours.enabled
        assert ours.action >= theirs.action
        assert ours.apply_to >= theirs.apply_to
        if isinstance(theirs.params, ThresholdParams):
            assert isinstance(ours.params, ThresholdParams)
            assert ours.params.min_score <= theirs.params.min_score


USERS = {"alice": "permissive", "carol": "strict", "dave": "mixed", "erin": "dormant"}
TEAMS = {
    "admins": "permissive",
    "contractors": "strict",
    "ops": "mixed",
    "idle": "dormant",
}
PAIRS = [(u, t) for u in USERS for t in TEAMS]


@pytest.mark.parametrize(("user_id", "team_id"), PAIRS)
def test_user_mapping_can_never_weaken_the_team_policy(
    resolver: PolicyResolver, user_id: str, team_id: str
) -> None:
    resolved = resolver.resolve(VerifiedIdentity(user_id=user_id, team_id=team_id))
    assert_at_least_as_strict(resolved, POLICIES[TEAMS[team_id]])
    assert_at_least_as_strict(resolved, POLICIES[USERS[user_id]])


def test_permissive_user_in_strict_team_gets_exactly_strict(
    resolver: PolicyResolver,
) -> None:
    resolved = resolver.resolve(
        VerifiedIdentity(user_id="alice", team_id="contractors")
    )
    strict = POLICIES["strict"]
    assert resolved.name == "permissive+strict"
    assert resolved.detectors == strict.detectors
    assert resolved.on_detector_error is strict.on_detector_error
    assert resolved.on_timeout is strict.on_timeout
    assert resolved.latency_budget_ms == strict.latency_budget_ms


def test_merge_takes_the_stricter_value_of_each_field() -> None:
    merged = merge_stricter(POLICIES["strict"], POLICIES["mixed"])
    rules, scored = merged.detectors["rules"], merged.detectors["scored"]
    assert (rules.enabled, rules.action) == (True, Action.BLOCK)
    assert rules.apply_to == {Role.USER, Role.TOOL, Role.SYSTEM}
    assert (scored.enabled, scored.action) == (True, Action.BLOCK)
    assert scored.apply_to == {Role.USER, Role.RESPONSE, Role.TOOL}
    assert scored.params == ThresholdParams(min_score=0.1)
    assert merged.on_detector_error is FailureMode.FAIL_CLOSED
    assert merged.on_timeout is FailureMode.FAIL_CLOSED
    assert merged.latency_budget_ms == 200


def test_merge_enables_a_detector_if_either_policy_enables_it() -> None:
    merged = merge_stricter(POLICIES["permissive"], POLICIES["default"])
    assert merged.detectors["rules"].enabled is True
    assert merged.detectors["rules"].apply_to == {Role.USER}


@pytest.mark.parametrize("disabled_first", [True, False])
def test_disabled_block_merged_with_enabled_allow_gives_enabled_allow(
    disabled_first: bool,
) -> None:
    # A disabled detector's action, roles and params have no effect, so they
    # must not leak into the merge.
    disabled = policy(
        "disabled",
        detector_config("block", ["system", "user", "tool"], enabled=False),
        detector_config(
            "block", ["user", "response"], enabled=False, params={"min_score": 0.0}
        ),
    )
    enabled = policy(
        "enabled",
        detector_config("allow", ["user"]),
        detector_config("allow", ["tool"], params={"min_score": 0.8}),
    )
    a, b = (disabled, enabled) if disabled_first else (enabled, disabled)
    merged = merge_stricter(a, b)
    rules, scored = merged.detectors["rules"], merged.detectors["scored"]
    assert (rules.enabled, rules.action, rules.apply_to) == (
        True,
        Action.ALLOW,
        {Role.USER},
    )
    assert (scored.enabled, scored.action, scored.apply_to) == (
        True,
        Action.ALLOW,
        {Role.TOOL},
    )
    assert scored.params == ThresholdParams(min_score=0.8)


def test_merge_of_two_disabled_detectors_is_disabled() -> None:
    merged = merge_stricter(POLICIES["permissive"], POLICIES["dormant"])
    rules = merged.detectors["rules"]
    assert rules.enabled is False
    # Neutral values: not taken from either side's inactive config.
    assert (rules.action, rules.apply_to) == (Action.ALLOW, frozenset())


def test_dormant_policy_adds_nothing_to_a_merge() -> None:
    # Every detector in "dormant" is disabled with the strictest settings;
    # merging it in changes no detector's effective behavior.
    for name in ("default", "strict", "permissive", "mixed"):
        merged = merge_stricter(POLICIES[name], POLICIES["dormant"])
        assert merged.detectors == POLICIES[name].detectors


def test_disabled_pii_params_do_not_count_in_merge() -> None:
    from llmfirewall.engine.detectors.builtin import PiiParams
    from llmfirewall.engine.engine import builtin_detectors

    detectors = builtin_detectors()

    def builtin_policy(name: str, pii: dict[str, Any]) -> Policy:
        return parse_policy(
            policy_data(
                {
                    "input_rules": detector_config("block", ["user"]),
                    "pii": pii,
                    "output_leak": detector_config("block", ["response"]),
                }
            ),
            name=name,
            detectors=detectors,
        )

    off = builtin_policy(
        "off",
        detector_config(
            "block",
            ["system", "user"],
            enabled=False,
            params={"entities": ["PERSON"], "min_score": 0.0},
        ),
    )
    on = builtin_policy(
        "on",
        detector_config(
            "redact",
            ["response"],
            params={"entities": ["EMAIL_ADDRESS"], "min_score": 0.5},
        ),
    )
    merged = merge_stricter(off, on).detectors["pii"]
    assert (merged.enabled, merged.action, merged.apply_to) == (
        True,
        Action.REDACT,
        {Role.RESPONSE},
    )
    assert merged.params == PiiParams.model_validate(
        {"entities": ["EMAIL_ADDRESS"], "min_score": 0.5}
    )


@pytest.mark.parametrize(("a", "b"), [(u, t) for u in POLICIES for t in POLICIES])
def test_merge_is_symmetric_apart_from_the_name(a: str, b: str) -> None:
    ab = merge_stricter(POLICIES[a], POLICIES[b])
    ba = merge_stricter(POLICIES[b], POLICIES[a])
    assert ab.model_dump(exclude={"name"}) == ba.model_dump(exclude={"name"})


def test_merge_of_shipped_default_and_strict_with_builtin_detectors() -> None:
    from llmfirewall.engine.detectors.builtin import PiiParams
    from llmfirewall.engine.engine import builtin_detectors

    detectors = builtin_detectors()
    default = load_policy_file(
        REPO_ROOT / "policies" / "default.yaml", detectors=detectors
    )
    strict = load_policy_file(
        REPO_ROOT / "policies" / "strict.yaml", detectors=detectors
    )
    merged = merge_stricter(default, strict)
    assert merged.detectors == merge_stricter(strict, default).detectors
    assert_at_least_as_strict(merged, default)
    assert_at_least_as_strict(merged, strict)
    params = merged.detectors["pii"].params
    assert isinstance(params, PiiParams)
    assert params.min_score == 0.0


def test_pii_params_merge_to_the_union_of_entities_and_the_lower_score() -> None:
    from llmfirewall.engine.detectors.builtin import PiiParams

    a = PiiParams.model_validate({"entities": ["PERSON"], "min_score": 0.6})
    b = PiiParams.model_validate({"entities": ["EMAIL_ADDRESS"], "min_score": 0.3})
    assert a.stricter(b) == PiiParams.model_validate(
        {"entities": ["EMAIL_ADDRESS", "PERSON"], "min_score": 0.3}
    )


def test_ids_are_matched_exactly(resolver: PolicyResolver) -> None:
    assert resolve(resolver, "Alice", None) == "default"
    assert resolve(resolver, " alice", None) == "default"


# Only verified fields are used --------------------------------------------


class RecordingAuth:
    """A stand-in for LiteLLM's UserAPIKeyAuth that records which attributes
    are read. Everything except user_id and team_id is client-influenced or
    deliberately unused, and is filled with values that would escalate."""

    def __init__(self, **fields: Any) -> None:
        object.__setattr__(self, "_fields", fields)
        object.__setattr__(self, "read", [])

    def __getattr__(self, name: str) -> Any:
        self.read.append(name)
        return self._fields.get(name)


def hostile_auth(user_id: Any = None, team_id: Any = None) -> RecordingAuth:
    return RecordingAuth(
        user_id=user_id,
        team_id=team_id,
        end_user_id="alice",  # from the request body's `user` field
        metadata={"llmfw_policy": "permissive", "policy": "permissive"},
        team_metadata={"llmfw_policy": "permissive"},
        key_alias="alice",
        team_alias="admins",
    )


def test_from_litellm_auth_reads_only_user_id_and_team_id() -> None:
    auth = hostile_auth(user_id="nobody", team_id="no-such-team")
    identity = VerifiedIdentity.from_litellm_auth(auth)
    assert identity == VerifiedIdentity(user_id="nobody", team_id="no-such-team")
    assert set(auth.read) <= {"user_id", "team_id"}


def test_from_litellm_auth_handles_missing_ids() -> None:
    identity = VerifiedIdentity.from_litellm_auth(SimpleNamespace())
    assert identity == VerifiedIdentity(user_id=None, team_id=None)


CLIENT_CONTROLLED = {
    # Where a policy name or a mapped id could show up from the request.
    "end_user_id from body `user`": {"end_user_id": "alice"},
    "request metadata": {"metadata": {"llmfw_policy": "permissive"}},
    "litellm_metadata": {"litellm_metadata": {"llmfw_policy": "permissive"}},
    "headers": {
        "request_headers": {
            "x-llmfw-policy": "permissive",
            "x-litellm-user-id": "alice",
            "x-litellm-team-id": "admins",
        }
    },
    "policy attribute": {"policy": "permissive", "llmfw_policy": "permissive"},
}


@pytest.mark.parametrize(
    "extra", CLIENT_CONTROLLED.values(), ids=CLIENT_CONTROLLED.keys()
)
def test_policy_name_in_request_metadata_or_headers_is_ignored(
    resolver: PolicyResolver, extra: dict[str, Any]
) -> None:
    auth = RecordingAuth(user_id="nobody", team_id=None, **extra)
    identity = VerifiedIdentity.from_litellm_auth(auth)
    assert resolver.resolve(identity).name == "default"
    assert set(auth.read) <= {"user_id", "team_id"}


def test_resolve_accepts_only_a_verified_identity() -> None:
    params = list(inspect.signature(PolicyResolver.resolve).parameters)
    assert params == ["self", "identity"]


def test_from_litellm_auth_accepts_only_the_auth_object() -> None:
    params = list(inspect.signature(VerifiedIdentity.from_litellm_auth).parameters)
    assert params == ["auth"]


def test_resolve_rejects_anything_but_a_verified_identity(
    resolver: PolicyResolver,
) -> None:
    for value in ({"user_id": "alice"}, "permissive", hostile_auth("alice")):
        with pytest.raises(TypeError):
            resolver.resolve(value)  # type: ignore[arg-type]


def test_verified_identity_is_frozen() -> None:
    identity = VerifiedIdentity(user_id="nobody", team_id=None)
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        identity.user_id = "alice"  # type: ignore[misc]


# Mapping validation -------------------------------------------------------


def with_changes(**changes: Any) -> dict[str, Any]:
    data = {**MAP_DATA, **changes}
    return {k: v for k, v in data.items() if v is not ...}


INVALID_MAPS = {
    "default missing": with_changes(default=...),
    "default is unknown policy": with_changes(default="nope"),
    "user maps to unknown policy": with_changes(users={"alice": "nope"}),
    "team maps to unknown policy": with_changes(teams={"t": "nope"}),
    "empty user id": with_changes(users={"": "strict"}),
    "empty team id": with_changes(teams={"": "strict"}),
    "version missing": with_changes(version=...),
    "version 2": with_changes(version=2),
    "unknown key": with_changes(headers={"x-llmfw-policy": "strict"}),
    "policy name not a string": with_changes(users={"alice": 1}),
}


@pytest.mark.parametrize("data", INVALID_MAPS.values(), ids=INVALID_MAPS.keys())
def test_invalid_identity_map_is_rejected(data: dict[str, Any]) -> None:
    with pytest.raises(PolicyError):
        parse_identity_map(data, POLICIES)


def test_users_and_teams_are_optional() -> None:
    identity_map = parse_identity_map({"version": 1, "default": "strict"}, POLICIES)
    assert isinstance(identity_map, IdentityMap)
    resolver = PolicyResolver(identity_map, POLICIES)
    assert resolve(resolver, "alice", "admins") == "strict"


def test_resolver_rejects_a_map_naming_policies_it_was_not_given() -> None:
    identity_map = parse_identity_map(MAP_DATA, POLICIES)
    with pytest.raises(PolicyError):
        PolicyResolver(identity_map, {"default": POLICIES["default"]})


def test_load_identity_map_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "identity.yaml"
    path.write_text("version: 1\ndefault: default\nteams:\n  contractors: strict\n")
    resolver = PolicyResolver(load_identity_map(path, POLICIES), POLICIES)
    assert resolve(resolver, None, "contractors") == "strict"


def test_load_identity_map_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "identity.yaml"
    path.write_text(
        "version: 1\ndefault: default\nusers:\n  carol: strict\n  carol: permissive\n"
    )
    with pytest.raises(PolicyError, match="duplicate"):
        load_identity_map(path, POLICIES)


def test_shipped_identity_map_maps_everyone_to_default() -> None:
    shipped = {"default": POLICIES["default"], "strict": POLICIES["strict"]}
    identity_map = load_identity_map(REPO_ROOT / "policies" / "identity.yaml", shipped)
    resolver = PolicyResolver(identity_map, shipped)
    assert resolve(resolver, "anyone", "any-team") == "default"
