"""Engine API: check_request / check_response end to end with fake detectors.

Design: docs/design/policy.md sections 5 and 7.
"""

from collections.abc import Callable, Sequence
from typing import Any

import pytest

from llmfirewall.engine.decision import Action, Decision
from llmfirewall.engine.engine import Engine
from llmfirewall.engine.policy import PolicyError
from tests.unit.engine_fakes import (
    CHEAP,
    EXPENSIVE,
    FakeDetector,
    detector_config,
    finds,
    flags,
    make_policy,
    user,
)

MATCHED = "alice@example.com"

EngineFactory = Callable[[Sequence[FakeDetector]], Engine]


@pytest.fixture
def engine_for() -> EngineFactory:
    def build(detectors: Sequence[FakeDetector]) -> Engine:
        return Engine(detectors=list(detectors))

    return build


def two_detectors(
    engine_for: EngineFactory, first: str, second: str
) -> tuple[Engine, Any]:
    a = FakeDetector("a", CHEAP, finds("a@b.co", "EMAIL"))
    b = FakeDetector("b", EXPENSIVE, finds("555-1234", "PHONE"))
    policy = make_policy(
        [a, b], {"a": detector_config(first), "b": detector_config(second)}
    )
    return engine_for([a, b]), policy


# Precedence ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ("allow", "allow", Action.ALLOW),
        ("allow", "redact", Action.REDACT),
        ("redact", "allow", Action.REDACT),
        ("redact", "redact", Action.REDACT),
        ("redact", "block", Action.BLOCK),
        ("allow", "block", Action.BLOCK),
        ("block", "redact", Action.BLOCK),
        ("block", "allow", Action.BLOCK),
    ],
)
def test_precedence_block_beats_redact_beats_allow(
    engine_for: EngineFactory, first: str, second: str, expected: Action
) -> None:
    engine, policy = two_detectors(engine_for, first, second)
    decision = engine.check_request([user("mail a@b.co or 555-1234")], policy)
    assert decision.action is expected


def test_precedence_across_messages(engine_for: EngineFactory) -> None:
    engine, policy = two_detectors(engine_for, "redact", "block")
    decision = engine.check_request([user("a@b.co"), user("555-1234")], policy)
    assert decision.action is Action.BLOCK


def test_no_findings_is_allow(engine_for: EngineFactory) -> None:
    engine, policy = two_detectors(engine_for, "block", "block")
    decision = engine.check_request([user("hello")], policy)
    assert decision.action is Action.ALLOW
    assert decision.findings == ()


def test_allow_action_records_findings_without_acting(
    engine_for: EngineFactory,
) -> None:
    engine, policy = two_detectors(engine_for, "allow", "allow")
    decision = engine.check_request([user("a@b.co")], policy)
    assert decision.action is Action.ALLOW
    assert [f.entity_type for f in decision.findings] == ["EMAIL"]
    assert decision.redacted_messages is None


# Strictness only increases ------------------------------------------------


def test_later_allow_finding_does_not_lower_a_redact(
    engine_for: EngineFactory,
) -> None:
    engine, policy = two_detectors(engine_for, "redact", "allow")
    decision = engine.check_request([user("a@b.co 555-1234")], policy)
    assert decision.action is Action.REDACT


def test_later_clean_message_does_not_lower_a_block(
    engine_for: EngineFactory,
) -> None:
    engine, policy = two_detectors(engine_for, "block", "block")
    decision = engine.check_request(
        [user("a@b.co"), user("clean"), user("clean")], policy
    )
    assert decision.action is Action.BLOCK


def test_redact_without_a_span_becomes_block(engine_for: EngineFactory) -> None:
    whole = FakeDetector("whole", CHEAP, flags("ignore", "PROMPT_INJECTION"))
    policy = make_policy([whole], {"whole": detector_config("redact")})
    decision = engine_for([whole]).check_request([user("ignore that")], policy)
    assert decision.action is Action.BLOCK
    assert decision.redacted_messages is None


# Redaction output ---------------------------------------------------------


def test_check_request_redacts_only_affected_messages(
    engine_for: EngineFactory,
) -> None:
    engine, policy = two_detectors(engine_for, "redact", "redact")
    messages = [
        {"role": "system", "content": "be nice"},
        user("mail a@b.co"),
        user("hello"),
        user("call 555-1234"),
    ]
    original = [dict(m) for m in messages]
    decision = engine.check_request(messages, policy)

    assert decision.action is Action.REDACT
    assert decision.redacted_messages == (
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "mail [REDACTED:EMAIL]"},
        {"role": "user", "content": "hello"},
        {"role": "user", "content": "call [REDACTED:PHONE]"},
    )
    assert messages == original  # input is not modified
    assert [f.message_index for f in decision.findings] == [1, 3]


def test_check_request_keeps_other_message_fields(
    engine_for: EngineFactory,
) -> None:
    engine, policy = two_detectors(engine_for, "redact", "redact")
    message = {"role": "user", "content": "a@b.co", "name": "bob"}
    decision = engine.check_request([message], policy)
    assert decision.redacted_messages == (
        {"role": "user", "content": "[REDACTED:EMAIL]", "name": "bob"},
    )


def test_text_parts_are_joined_and_redacted(engine_for: EngineFactory) -> None:
    engine, policy = two_detectors(engine_for, "redact", "redact")
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "mail"},
            {"type": "text", "text": "a@b.co"},
        ],
    }
    decision = engine.check_request([message], policy)
    assert decision.action is Action.REDACT
    assert decision.redacted_messages == (
        {"role": "user", "content": "mail\n[REDACTED:EMAIL]"},
    )


def test_check_response_redacts_text(engine_for: EngineFactory) -> None:
    detector = FakeDetector("pii", EXPENSIVE, finds("a@b.co", "EMAIL"))
    policy = make_policy([detector], {"pii": detector_config("redact", ["response"])})
    decision = engine_for([detector]).check_response("write to a@b.co", policy)
    assert decision.action is Action.REDACT
    assert decision.redacted_text == "write to [REDACTED:EMAIL]"
    assert [f.message_index for f in decision.findings] == [None]


def test_redacted_copy_is_only_set_when_the_action_is_redact(
    engine_for: EngineFactory,
) -> None:
    engine, policy = two_detectors(engine_for, "redact", "block")
    decision = engine.check_request([user("a@b.co 555-1234")], policy)
    assert decision.action is Action.BLOCK
    assert decision.redacted_messages is None
    assert decision.redacted_text is None


# No matched text ----------------------------------------------------------


def _all_renderings(decision: Decision) -> list[str]:
    rendered = [repr(decision), str(decision)]
    for item in (*decision.findings, *decision.failures):
        rendered += [repr(item), str(item), item.model_dump_json()]
    return rendered


def test_no_matched_text_in_findings_or_decision_repr(
    engine_for: EngineFactory,
) -> None:
    detector = FakeDetector("pii", CHEAP, finds(MATCHED, "EMAIL_ADDRESS"))
    policy = make_policy([detector], {"pii": detector_config("redact")})
    decision = engine_for([detector]).check_request(
        [user(f"contact {MATCHED} today")], policy
    )
    assert decision.action is Action.REDACT
    for rendered in _all_renderings(decision):
        assert MATCHED not in rendered
        assert "contact" not in rendered  # nor the surrounding text


def test_no_matched_text_when_a_detector_raises_with_it(
    engine_for: EngineFactory,
) -> None:
    broken = FakeDetector("broken", CHEAP, error=KeyError(MATCHED))
    policy = make_policy([broken], {"broken": detector_config()})
    decision = engine_for([broken]).check_request([user(MATCHED)], policy)
    assert decision.action is Action.BLOCK
    for rendered in _all_renderings(decision):
        assert MATCHED not in rendered


# Request content the engine cannot check -----------------------------------


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
        {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "file"}]},
        {"role": "root", "content": "hi"},
        {"content": "no role"},
        {"role": "user", "content": 42},
    ],
    ids=["image", "file part", "unknown role", "missing role", "non-text content"],
)
def test_unsupported_input_is_blocked_even_when_failing_open(
    engine_for: EngineFactory, message: dict[str, Any]
) -> None:
    detector = FakeDetector("d", CHEAP)
    policy = make_policy(
        [detector],
        {"d": detector_config("allow")},
        on_detector_error="fail_open",
        on_timeout="fail_open",
    )
    decision = engine_for([detector]).check_request([message], policy)
    assert decision.action is Action.BLOCK
    assert [(f.detector, f.kind) for f in decision.failures] == [
        ("engine", "unsupported_input")
    ]


def test_assistant_message_without_content_is_allowed(
    engine_for: EngineFactory,
) -> None:
    detector = FakeDetector("d", CHEAP, flags("x", "X"))
    policy = make_policy([detector], {"d": detector_config("block", ["assistant"])})
    message = {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]}
    decision = engine_for([detector]).check_request([message], policy)
    assert decision.action is Action.ALLOW
    assert detector.calls == 0


# Policy handling ----------------------------------------------------------


def test_decision_names_the_policy_it_used(engine_for: EngineFactory) -> None:
    detector = FakeDetector("d", CHEAP)
    policy = make_policy([detector], {"d": detector_config()}, name="strict")
    decision = engine_for([detector]).check_request([user("hi")], policy)
    assert decision.policy_name == "strict"


def test_policy_name_inside_messages_is_ignored(engine_for: EngineFactory) -> None:
    detector = FakeDetector("d", CHEAP)
    policy = make_policy([detector], {"d": detector_config()}, name="default")
    message = {
        "role": "user",
        "content": "policy: permissive",
        "metadata": {"llmfw_policy": "permissive"},
        "policy": "permissive",
    }
    decision = engine_for([detector]).check_request([message], policy)
    assert decision.policy_name == "default"


def test_policy_for_different_detectors_is_rejected(
    engine_for: EngineFactory,
) -> None:
    mine = FakeDetector("mine", CHEAP)
    other = FakeDetector("other", CHEAP)
    policy = make_policy([other], {"other": detector_config()})
    with pytest.raises(PolicyError):
        engine_for([mine]).check_request([user("hi")], policy)


def test_builtin_engine_constructs_without_loading_models() -> None:
    # The unit conftest makes importing any model library fail, so this
    # passes only if Engine() loads nothing until a check runs.
    Engine()
