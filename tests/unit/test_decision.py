"""Decision rules: precedence, strictness, redaction, and Findings without text.

Design: docs/design/policy.md sections 4 and 5.
"""

import itertools

import pytest
from pydantic import ValidationError

from llmfirewall.engine.decision import (
    Action,
    Decision,
    DetectorFailure,
    Finding,
    Span,
    Verdict,
    combine,
    redact_text,
)

MATCHED = "alice@example.com"


def finding(
    start: int | None,
    end: int | None,
    entity_type: str = "EMAIL_ADDRESS",
    score: float = 1.0,
) -> Finding:
    span = None if start is None or end is None else Span(start=start, end=end)
    return Finding(
        detector="pii",
        entity_type=entity_type,
        score=score,
        message_index=0,
        span=span,
        duration_ms=1.0,
    )


# Precedence ---------------------------------------------------------------


def test_actions_are_ordered_allow_redact_block() -> None:
    assert Action.ALLOW < Action.REDACT < Action.BLOCK


@pytest.mark.parametrize(
    ("actions", "expected"),
    [
        ([], Action.ALLOW),
        ([Action.ALLOW], Action.ALLOW),
        ([Action.ALLOW, Action.REDACT], Action.REDACT),
        ([Action.REDACT, Action.ALLOW], Action.REDACT),
        ([Action.REDACT, Action.BLOCK], Action.BLOCK),
        ([Action.BLOCK, Action.REDACT, Action.ALLOW], Action.BLOCK),
        ([Action.ALLOW, Action.ALLOW, Action.BLOCK], Action.BLOCK),
    ],
)
def test_combine_block_beats_redact_beats_allow(
    actions: list[Action], expected: Action
) -> None:
    assert combine(actions) is expected


@pytest.mark.parametrize("order", list(itertools.permutations(list(Action))))
def test_combine_does_not_depend_on_order(order: tuple[Action, ...]) -> None:
    assert combine(order) is Action.BLOCK


# Strictness only increases ------------------------------------------------


def test_verdict_starts_at_allow() -> None:
    assert Verdict().action is Action.ALLOW


@pytest.mark.parametrize("sequence", list(itertools.product(list(Action), repeat=4)))
def test_verdict_never_decreases(sequence: tuple[Action, ...]) -> None:
    verdict = Verdict()
    previous = verdict.action
    for action in sequence:
        verdict.escalate(action)
        assert verdict.action >= previous
        assert verdict.action >= action
        previous = verdict.action
    assert verdict.action is max(sequence)


def test_verdict_has_no_way_to_lower_or_set_the_action() -> None:
    verdict = Verdict()
    verdict.escalate(Action.BLOCK)
    with pytest.raises(AttributeError):
        verdict.action = Action.ALLOW  # type: ignore[misc]
    public = {n for n in dir(verdict) if not n.startswith("_")}
    assert public == {"action", "escalate"}


def test_decision_is_frozen() -> None:
    decision = Decision(
        action=Action.BLOCK,
        policy_name="default",
        findings=(),
        failures=(),
        skipped=(),
    )
    with pytest.raises(ValidationError):
        decision.action = Action.ALLOW  # type: ignore[misc]


# Redaction ----------------------------------------------------------------


def test_redact_replaces_each_span_with_a_typed_placeholder() -> None:
    text = "mail a@b.co or call 555-1234"
    findings = [
        finding(5, 11, "EMAIL_ADDRESS"),
        finding(20, 28, "PHONE_NUMBER"),
    ]
    assert redact_text(text, findings) == (
        "mail [REDACTED:EMAIL_ADDRESS] or call [REDACTED:PHONE_NUMBER]"
    )


def test_redact_order_of_findings_does_not_matter() -> None:
    text = "mail a@b.co or call 555-1234"
    findings = [finding(20, 28, "PHONE_NUMBER"), finding(5, 11, "EMAIL_ADDRESS")]
    assert redact_text(text, findings) == (
        "mail [REDACTED:EMAIL_ADDRESS] or call [REDACTED:PHONE_NUMBER]"
    )


def test_redact_overlapping_spans_become_one_placeholder() -> None:
    # "John Smith" (PERSON 0-10) overlaps "Smith@x.io" (EMAIL 5-15).
    text = "John Smith@x.io wrote"
    findings = [
        finding(0, 10, "PERSON", score=0.85),
        finding(5, 15, "EMAIL_ADDRESS", score=1.0),
    ]
    assert redact_text(text, findings) == "[REDACTED:EMAIL_ADDRESS] wrote"


def test_redact_nested_span_is_covered_by_the_outer_one() -> None:
    text = "key sk-abcdef1234567890 end"
    findings = [
        finding(4, 23, "API_KEY", score=1.0),
        finding(7, 13, "PERSON", score=0.4),
    ]
    assert redact_text(text, findings) == "key [REDACTED:API_KEY] end"


def test_redact_chain_of_overlaps_merges_transitively() -> None:
    # 0-4 overlaps 3-7, which overlaps 6-10; 0-4 and 6-10 do not overlap
    # directly but end up in one group.
    text = "abcdefghijXYZ"
    findings = [
        finding(0, 4, "A", score=0.5),
        finding(3, 7, "B", score=0.9),
        finding(6, 10, "C", score=0.5),
    ]
    assert redact_text(text, findings) == "[REDACTED:B]XYZ"


def test_redact_identical_spans_tie_on_score_goes_to_alphabetical_type() -> None:
    text = "x 4111111111111111 y"
    findings = [
        finding(2, 18, "US_SSN", score=0.9),
        finding(2, 18, "CREDIT_CARD", score=0.9),
    ]
    assert redact_text(text, findings) == "x [REDACTED:CREDIT_CARD] y"


def test_redact_touching_spans_are_not_merged() -> None:
    text = "abcdef"
    findings = [finding(0, 3, "A"), finding(3, 6, "B")]
    assert redact_text(text, findings) == "[REDACTED:A][REDACTED:B]"


def test_redact_with_no_findings_returns_text_unchanged() -> None:
    assert redact_text("nothing here", []) == "nothing here"


def test_redact_rejects_a_span_past_the_end_of_the_text() -> None:
    with pytest.raises(ValueError):
        redact_text("short", [finding(2, 50)])


def test_redact_rejects_a_finding_without_a_span() -> None:
    with pytest.raises(ValueError):
        redact_text("text", [finding(None, None)])


# Spans and Findings -------------------------------------------------------


@pytest.mark.parametrize(("start", "end"), [(-1, 3), (3, 3), (5, 2)])
def test_span_must_be_non_empty_and_non_negative(start: int, end: int) -> None:
    with pytest.raises(ValidationError):
        Span(start=start, end=end)


@pytest.mark.parametrize("score", [-0.1, 1.1])
def test_finding_score_is_between_0_and_1(score: float) -> None:
    with pytest.raises(ValidationError):
        finding(0, 3, score=score)


def test_finding_has_exactly_the_designed_fields() -> None:
    # Pinned on purpose: adding a field (for example one holding text) must
    # be a deliberate change to this test.
    assert set(Finding.model_fields) == {
        "detector",
        "entity_type",
        "score",
        "message_index",
        "span",
        "duration_ms",
    }


def test_finding_rejects_extra_fields_such_as_text() -> None:
    with pytest.raises(ValidationError):
        Finding(
            detector="pii",
            entity_type="EMAIL_ADDRESS",
            score=1.0,
            message_index=0,
            span=Span(start=0, end=len(MATCHED)),
            duration_ms=1.0,
            text=MATCHED,  # type: ignore[call-arg]
        )


def test_finding_repr_str_and_json_contain_no_matched_text() -> None:
    f = finding(0, len(MATCHED))
    for rendered in (repr(f), str(f), f.model_dump_json(), str(f.model_dump())):
        assert MATCHED not in rendered
        assert "alice" not in rendered


def test_finding_is_frozen() -> None:
    f = finding(0, 3)
    with pytest.raises(ValidationError):
        f.entity_type = "OTHER"  # type: ignore[misc]


def test_detector_failure_has_no_field_for_the_exception_message() -> None:
    assert set(DetectorFailure.model_fields) == {
        "detector",
        "kind",
        "error_type",
        "message_index",
    }
