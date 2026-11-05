"""Policy schema validation and loading.

Design: docs/design/policy.md section 2.
"""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from llmfirewall.engine.decision import Action
from llmfirewall.engine.policy import (
    FailureMode,
    PolicyError,
    Role,
    load_policies,
    load_policy_file,
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

RULES = FakeDetector("rules", CHEAP)
SCORED = FakeDetector("scored", EXPENSIVE, params_model=ThresholdParams)
DETECTORS = [RULES, SCORED]


def valid_data() -> dict[str, Any]:
    return policy_data(
        {
            "rules": detector_config("block", ["user"]),
            "scored": detector_config(
                "redact", ["user", "response"], params={"min_score": 0.5}
            ),
        }
    )


def parse(data: dict[str, Any]) -> Any:
    return parse_policy(data, name="test", detectors=DETECTORS)


# Valid policies -----------------------------------------------------------


def test_valid_policy_parses() -> None:
    policy = parse(valid_data())
    assert policy.name == "test"
    assert policy.version == 1
    assert policy.latency_budget_ms == 1000
    assert policy.detectors["rules"].action is Action.BLOCK
    assert policy.detectors["rules"].apply_to == frozenset({Role.USER})
    assert policy.detectors["scored"].action is Action.REDACT
    assert policy.detectors["scored"].apply_to == frozenset({Role.USER, Role.RESPONSE})
    params = policy.detectors["scored"].params
    assert isinstance(params, ThresholdParams)
    assert params.min_score == 0.5


def test_failure_modes_default_to_fail_closed() -> None:
    data = valid_data()
    del data["on_detector_error"]
    del data["on_timeout"]
    policy = parse(data)
    assert policy.on_detector_error is FailureMode.FAIL_CLOSED
    assert policy.on_timeout is FailureMode.FAIL_CLOSED


def test_params_default_to_the_params_model_defaults() -> None:
    data = valid_data()
    del data["detectors"]["scored"]["params"]
    params = parse(data).detectors["scored"].params
    assert params == ThresholdParams()


def test_disabled_detector_may_have_empty_apply_to() -> None:
    data = valid_data()
    data["detectors"]["rules"] = detector_config("block", [], enabled=False)
    assert parse(data).detectors["rules"].enabled is False


def test_policy_is_frozen() -> None:
    policy = parse(valid_data())
    with pytest.raises(ValidationError):
        policy.on_timeout = FailureMode.FAIL_OPEN  # type: ignore[misc]


# Validation errors --------------------------------------------------------


def _set(path: tuple[str, ...], value: Any) -> Any:
    def mutate(data: dict[str, Any]) -> None:
        target = data
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


def _delete(path: tuple[str, ...]) -> Any:
    def mutate(data: dict[str, Any]) -> None:
        target = data
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]

    return mutate


INVALID = {
    "version missing": _delete(("version",)),
    "version 2": _set(("version",), 2),
    "version as string": _set(("version",), "1"),
    "unknown top-level key": _set(("mode",), "permissive"),
    "on_detector_error unknown": _set(("on_detector_error",), "ignore"),
    "on_timeout unknown": _set(("on_timeout",), "maybe"),
    "latency budget missing": _delete(("latency_budget_ms",)),
    "latency budget zero": _set(("latency_budget_ms",), 0),
    "latency budget negative": _set(("latency_budget_ms",), -5),
    "detectors missing": _delete(("detectors",)),
    "unknown detector": _set(("detectors", "nope"), detector_config()),
    "registered detector missing": _delete(("detectors", "scored")),
    "action unknown": _set(("detectors", "rules", "action"), "deny"),
    "action uppercase": _set(("detectors", "rules", "action"), "BLOCK"),
    "enabled missing": _delete(("detectors", "rules", "enabled")),
    "enabled as string": _set(("detectors", "rules", "enabled"), "true"),
    "action missing": _delete(("detectors", "rules", "action")),
    "apply_to missing": _delete(("detectors", "rules", "apply_to")),
    "apply_to unknown role": _set(("detectors", "rules", "apply_to"), ["admin"]),
    "apply_to empty when enabled": _set(("detectors", "rules", "apply_to"), []),
    "apply_to as string": _set(("detectors", "rules", "apply_to"), "user"),
    "extra key in detector config": _set(("detectors", "rules", "threshold"), 0.5),
    "params on detector without params": _set(
        ("detectors", "rules", "params"), {"x": 1}
    ),
    "unknown param": _set(("detectors", "scored", "params"), {"min_scor": 0.5}),
    "param out of range": _set(("detectors", "scored", "params"), {"min_score": 2.0}),
    "param wrong type": _set(("detectors", "scored", "params"), {"min_score": "high"}),
}


@pytest.mark.parametrize("mutate", INVALID.values(), ids=INVALID.keys())
def test_invalid_policy_is_rejected(mutate: Any) -> None:
    data = valid_data()
    mutate(data)
    with pytest.raises(PolicyError):
        parse(data)


def test_policy_that_is_not_a_mapping_is_rejected() -> None:
    with pytest.raises(PolicyError):
        parse_policy(["version", 1], name="test", detectors=DETECTORS)  # type: ignore[arg-type]


def test_two_detectors_with_the_same_name_are_rejected() -> None:
    with pytest.raises(PolicyError):
        parse_policy(
            valid_data(),
            name="test",
            detectors=[RULES, FakeDetector("rules", EXPENSIVE), SCORED],
        )


def test_policy_error_names_the_offending_field() -> None:
    data = valid_data()
    data["detectors"]["rules"]["action"] = "deny"
    with pytest.raises(PolicyError, match="rules") as excinfo:
        parse(data)
    assert "action" in str(excinfo.value)


# Loading YAML -------------------------------------------------------------

VALID_YAML = """\
version: 1
latency_budget_ms: 500
detectors:
  rules:
    enabled: true
    action: block
    apply_to: [user]
  scored:
    enabled: true
    action: redact
    apply_to: [response]
    params:
      min_score: 0.3
"""


def test_load_policy_file_takes_its_name_from_the_file(tmp_path: Path) -> None:
    path = tmp_path / "team-a.yaml"
    path.write_text(VALID_YAML)
    policy = load_policy_file(path, detectors=DETECTORS)
    assert policy.name == "team-a"
    assert policy.latency_budget_ms == 500


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    # Plain yaml.safe_load keeps the last value silently, which would turn
    # this block into an allow.
    path = tmp_path / "dup.yaml"
    path.write_text(
        VALID_YAML.replace(
            "    action: block\n", "    action: block\n    action: allow\n"
        )
    )
    with pytest.raises(PolicyError, match="duplicate"):
        load_policy_file(path, detectors=DETECTORS)


def test_yaml_that_is_not_a_mapping_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- version: 1\n")
    with pytest.raises(PolicyError):
        load_policy_file(path, detectors=DETECTORS)


def test_malformed_yaml_is_a_policy_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("version: 1\ndetectors: [unclosed\n")
    with pytest.raises(PolicyError):
        load_policy_file(path, detectors=DETECTORS)


def test_yaml_tags_that_build_objects_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "tag.yaml"
    path.write_text("version: !!python/object/apply:os.getcwd []\n")
    with pytest.raises(PolicyError):
        load_policy_file(path, detectors=DETECTORS)


def test_load_policies_reads_every_yaml_file_except_identity(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text(VALID_YAML)
    (tmp_path / "b.yaml").write_text(VALID_YAML)
    (tmp_path / "identity.yaml").write_text("version: 1\ndefault: a\n")
    (tmp_path / "notes.txt").write_text("not a policy")
    policies = load_policies(tmp_path, detectors=DETECTORS)
    assert set(policies) == {"a", "b"}


def test_load_policies_fails_if_any_file_is_invalid(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text(VALID_YAML)
    (tmp_path / "b.yaml").write_text(VALID_YAML.replace("block", "deny"))
    with pytest.raises(PolicyError, match="b"):
        load_policies(tmp_path, detectors=DETECTORS)


# Shipped policies ---------------------------------------------------------


@pytest.mark.parametrize("name", ["default", "strict"])
def test_shipped_policy_is_valid_for_the_builtin_detectors(name: str) -> None:
    from llmfirewall.engine.engine import builtin_detectors

    policy = load_policy_file(
        REPO_ROOT / "policies" / f"{name}.yaml", detectors=builtin_detectors()
    )
    assert policy.name == name
    assert policy.on_detector_error is FailureMode.FAIL_CLOSED
    assert policy.on_timeout is FailureMode.FAIL_CLOSED
    assert set(policy.detectors) == {"input_rules", "pii", "output_leak"}
