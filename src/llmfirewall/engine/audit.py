"""Audit events: types, offsets, counts and timings. Never raw text.

One event per decision (a request, a non-streaming response, or a whole
stream). An event is built only from a Decision's findings, failures,
skipped detectors and durations, plus identity that LiteLLM verified:
- team_id as LiteLLM reports it (an id the proxy admin assigned);
- user_id only as HMAC-SHA256(LLMFW_AUDIT_HMAC_KEY, user_id), never raw.

No field can hold request or response text. Every model is frozen with
extra="forbid", and tests pin the field sets, so adding a field is a
deliberate change. Strings that come from outside the engine are limited to
values that cannot carry free text: the call id must parse as a UUID, and the
call type must be one LiteLLM defines.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import uuid
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from llmfirewall.engine.decision import Action, Decision

AUDIT_LOGGER = "llmfirewall.audit"

_FROZEN = ConfigDict(frozen=True, extra="forbid")

Direction = Literal["request", "response", "stream"]
Reason = Literal[
    "decision",  # the engine decided (allow, redact or block)
    "unrestricted_key",  # the key has no allowed_routes: blocked before any check
    "unsupported_call_type",  # not a chat completion: blocked before any check
    "unsupported_response",  # a response the adapter cannot read: blocked
    "internal_error",  # the adapter or engine raised: blocked
]


_ACTION_NAMES: dict[Action, Literal["allow", "redact", "block"]] = {
    Action.ALLOW: "allow",
    Action.REDACT: "redact",
    Action.BLOCK: "block",
}


class AuditFinding(BaseModel):
    model_config = _FROZEN

    detector: str
    entity_type: str
    message_index: int | None
    # Character offsets [start, end) in the checked text; None when the
    # detector flags the whole text. For a stream, offsets are in the whole
    # response as generated.
    start: int | None
    end: int | None


class AuditFailure(BaseModel):
    model_config = _FROZEN

    detector: str
    kind: str
    error_type: str | None
    message_index: int | None


class StreamStop(BaseModel):
    """Why and where a stream was stopped before the model finished."""

    model_config = _FROZEN

    reason: Literal["content_filter"]
    choice_index: int
    released_chars: int = Field(ge=0)  # characters the client had received
    withheld_chars: int = Field(ge=0)  # held back and never sent


class AuditEvent(BaseModel):
    model_config = _FROZEN

    version: Literal[1] = 1
    event_id: str
    timestamp: str  # ISO 8601, UTC
    call_id: str | None  # LiteLLM's call id, kept only if it is a UUID
    call_type: str | None
    direction: Direction
    reason: Reason
    action: Literal["allow", "redact", "block"]
    policy_name: str | None
    team_id: str | None
    user_id_hmac: str | None
    findings: tuple[AuditFinding, ...]
    counts: dict[str, int]  # entity_type -> number of findings
    detector_ms: dict[str, float]  # detector -> total time, all checks summed
    failures: tuple[AuditFailure, ...]
    skipped: tuple[str, ...]
    checks: int = Field(ge=0)  # engine calls behind this event
    stream_stop: StreamStop | None = None


class AuditSink(Protocol):
    def emit(self, event: AuditEvent) -> None: ...


class LoggingSink:
    """Writes each event as one JSON line to the llmfirewall.audit logger."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(AUDIT_LOGGER)

    def emit(self, event: AuditEvent) -> None:
        self._logger.info(event.model_dump_json())


class MissingAuditKey(ValueError):
    """LLMFW_AUDIT_HMAC_KEY is not set. The adapter refuses to start
    rather than log raw user ids or no user at all."""


def user_id_hmac(key: bytes, user_id: str | None) -> str | None:
    if user_id is None:
        return None
    return hmac.new(key, user_id.encode("utf-8"), hashlib.sha256).hexdigest()


def safe_call_id(value: object) -> str | None:
    """The value as a canonical UUID string, or None. Anything else may be
    client-chosen text, so it is dropped."""
    if not isinstance(value, str):
        return None
    try:
        return str(uuid.UUID(value))
    except ValueError:
        return None


def _finding(
    detector: str,
    entity_type: str,
    message_index: int | None,
    start: int | None,
    end: int | None,
) -> AuditFinding:
    return AuditFinding(
        detector=detector,
        entity_type=entity_type,
        message_index=message_index,
        start=start,
        end=end,
    )


def findings_of(decision: Decision, offset: int = 0) -> list[AuditFinding]:
    """The decision's findings, with spans shifted by `offset`."""
    return [
        _finding(
            f.detector,
            f.entity_type,
            f.message_index,
            None if f.span is None else f.span.start + offset,
            None if f.span is None else f.span.end + offset,
        )
        for f in decision.findings
    ]


def failures_of(decision: Decision) -> list[AuditFailure]:
    return [
        AuditFailure(
            detector=f.detector,
            kind=f.kind,
            error_type=f.error_type,
            message_index=f.message_index,
        )
        for f in decision.failures
    ]


def sum_durations(decisions: Iterable[Decision]) -> dict[str, float]:
    total: dict[str, float] = {}
    for decision in decisions:
        for name, ms in decision.durations_ms.items():
            total[name] = total.get(name, 0.0) + ms
    return {name: round(ms, 3) for name, ms in sorted(total.items())}


def build_event(
    *,
    direction: Direction,
    reason: Reason,
    action: Action,
    policy_name: str | None,
    team_id: str | None,
    user_hmac: str | None,
    call_id: object,
    call_type: str | None,
    findings: Iterable[AuditFinding] = (),
    failures: Iterable[AuditFailure] = (),
    skipped: Iterable[str] = (),
    decisions: Iterable[Decision] = (),
    stream_stop: StreamStop | None = None,
) -> AuditEvent:
    decisions = list(decisions)
    # A stream re-scans text, so the same finding can appear in several
    # windows; keep each (detector, type, place) once.
    unique = list(dict.fromkeys(findings))
    return AuditEvent(
        event_id=str(uuid.uuid4()),
        timestamp=datetime.now(UTC).isoformat(),
        call_id=safe_call_id(call_id),
        call_type=call_type,
        direction=direction,
        reason=reason,
        action=_ACTION_NAMES[action],
        policy_name=policy_name,
        team_id=team_id,
        user_id_hmac=user_hmac,
        findings=tuple(unique),
        counts=dict(sorted(Counter(f.entity_type for f in unique).items())),
        detector_ms=sum_durations(decisions),
        failures=tuple(dict.fromkeys(failures)),
        skipped=tuple(sorted(set(skipped))),
        checks=len(decisions),
        stream_stop=stream_stop,
    )
