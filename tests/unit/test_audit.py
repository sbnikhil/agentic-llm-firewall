"""engine.audit: the event shape is pinned and can never carry text."""

from __future__ import annotations

import hashlib
import hmac

from llmfirewall.engine.audit import (
    AuditEvent,
    AuditFailure,
    AuditFinding,
    StreamStop,
    build_event,
    findings_of,
    safe_call_id,
    user_id_hmac,
)
from llmfirewall.engine.decision import Action
from llmfirewall.engine.engine import Engine
from tests.fake_secrets import AUDIT_API_KEY
from tests.unit.engine_fakes import (
    EXPENSIVE,
    FakeDetector,
    detector_config,
    finds,
    make_policy,
    user,
)


def test_event_field_set_is_pinned() -> None:
    assert set(AuditEvent.model_fields) == {
        "version",
        "event_id",
        "timestamp",
        "call_id",
        "call_type",
        "direction",
        "reason",
        "action",
        "policy_name",
        "team_id",
        "user_id_hmac",
        "findings",
        "counts",
        "detector_ms",
        "failures",
        "skipped",
        "checks",
        "stream_stop",
    }
    assert set(AuditFinding.model_fields) == {
        "detector",
        "entity_type",
        "message_index",
        "start",
        "end",
    }
    assert set(AuditFailure.model_fields) == {
        "detector",
        "kind",
        "error_type",
        "message_index",
    }
    assert set(StreamStop.model_fields) == {
        "reason",
        "choice_index",
        "released_chars",
        "withheld_chars",
    }


def test_models_forbid_extra_fields() -> None:
    for model in (AuditEvent, AuditFinding, AuditFailure, StreamStop):
        assert model.model_config.get("extra") == "forbid"
        assert model.model_config.get("frozen") is True


def test_user_id_hmac_is_hmac_sha256() -> None:
    expected = hmac.new(b"k", b"alice", hashlib.sha256).hexdigest()
    assert user_id_hmac(b"k", "alice") == expected
    assert user_id_hmac(b"k", None) is None
    assert user_id_hmac(b"other", "alice") != expected


def test_call_id_must_be_a_uuid() -> None:
    assert (
        safe_call_id("1B4E28BA-2FA1-11D2-883F-0016D3CCA427")
        == "1b4e28ba-2fa1-11d2-883f-0016d3cca427"
    )
    for bad in ("my secret prompt", "", None, 12, "1b4e28ba"):
        assert safe_call_id(bad) is None


def test_event_from_a_decision_has_offsets_counts_and_timings_only() -> None:
    secret = AUDIT_API_KEY
    det = FakeDetector("secret", EXPENSIVE, finds(secret, "API_KEY"))
    policy = make_policy([det], {"secret": detector_config("redact", ["user"])})
    decision = Engine([det]).check_request([user(f"a {secret} b {secret}")], policy)
    event = build_event(
        direction="request",
        reason="decision",
        action=decision.action,
        policy_name=policy.name,
        team_id="t",
        user_hmac=user_id_hmac(b"k", "u"),
        call_id=None,
        call_type="completion",
        findings=findings_of(decision),
        decisions=[decision],
    )
    raw = event.model_dump_json()
    assert secret not in raw
    assert event.action == "redact"
    assert event.counts == {"API_KEY": 2}
    assert [(f.start, f.end) for f in event.findings] == [
        (2, 2 + len(secret)),
        (5 + len(secret), 5 + 2 * len(secret)),
    ]
    assert set(event.detector_ms) == {"secret"}
    assert event.checks == 1


def test_duplicate_findings_from_overlapping_windows_are_counted_once() -> None:
    f = AuditFinding(detector="d", entity_type="X", message_index=None, start=1, end=3)
    event = build_event(
        direction="stream",
        reason="decision",
        action=Action.REDACT,
        policy_name="p",
        team_id=None,
        user_hmac=None,
        call_id=None,
        call_type=None,
        findings=[f, f, f],
    )
    assert event.findings == (f,)
    assert event.counts == {"X": 1}
