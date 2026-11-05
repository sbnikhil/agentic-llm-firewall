"""Findings, actions and decisions; redaction with typed placeholders.

Design: docs/design/policy.md sections 4 and 5. Nothing here stores matched
text: a Finding records where and what, never the characters themselves.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import IntEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Action(IntEnum):
    """Ordered by strictness, so max() picks the stricter action."""

    ALLOW = 0
    REDACT = 1
    BLOCK = 2


def combine(actions: Iterable[Action]) -> Action:
    """Block beats redact beats allow. No actions means allow."""
    return max(actions, default=Action.ALLOW)


class Verdict:
    """The running action of one check. It can only go up."""

    __slots__ = ("_action",)

    def __init__(self) -> None:
        self._action = Action.ALLOW

    @property
    def action(self) -> Action:
        return self._action

    def escalate(self, action: Action) -> None:
        self._action = max(self._action, action)


_FROZEN = ConfigDict(frozen=True, extra="forbid")


class Span(BaseModel):
    """Character offsets [start, end) in Python str indices."""

    model_config = _FROZEN

    start: int = Field(ge=0)
    end: int

    @model_validator(mode="after")
    def _non_empty(self) -> Span:
        if self.end <= self.start:
            raise ValueError("span end must be greater than start")
        return self


class Finding(BaseModel):
    model_config = _FROZEN

    detector: str
    entity_type: str
    score: float = Field(ge=0.0, le=1.0)
    message_index: int | None
    span: Span | None
    duration_ms: float = Field(ge=0.0)


FailureKind = Literal["error", "timeout", "invalid_output", "unsupported_input"]


class DetectorFailure(BaseModel):
    """A check that could not run properly. Holds the exception's class name
    only: exception messages often contain the input."""

    model_config = _FROZEN

    detector: str
    kind: FailureKind
    error_type: str | None
    message_index: int | None


class Decision(BaseModel):
    model_config = _FROZEN

    action: Action
    policy_name: str
    findings: tuple[Finding, ...]
    failures: tuple[DetectorFailure, ...]
    skipped: tuple[str, ...]
    # Time per detector that ran, in ms (audit only).
    durations_ms: dict[str, float] = Field(default_factory=dict)
    # The content to forward when action is REDACT. Excluded from repr so
    # that logging a decision never logs text.
    redacted_messages: tuple[dict[str, Any], ...] | None = Field(
        default=None, repr=False
    )
    redacted_text: str | None = Field(default=None, repr=False)


def placeholder(entity_type: str) -> str:
    return f"[REDACTED:{entity_type}]"


def redact_text(text: str, findings: Sequence[Finding]) -> str:
    """Replace each finding's span with a typed placeholder.

    Overlapping spans (sharing at least one character) merge into one span
    covering all of them, labelled with the highest-scoring finding's type;
    ties go to the alphabetically first type. Touching spans stay separate.
    """
    spans: list[tuple[int, int, Finding]] = []
    for finding in findings:
        if finding.span is None:
            raise ValueError("cannot redact a finding without a span")
        if finding.span.end > len(text):
            raise ValueError("span is outside the text")
        spans.append((finding.span.start, finding.span.end, finding))
    spans.sort(key=lambda s: (s[0], s[1]))

    groups: list[tuple[int, int, list[Finding]]] = []
    for start, end, finding in spans:
        if groups and start < groups[-1][1]:
            g_start, g_end, members = groups[-1]
            groups[-1] = (g_start, max(g_end, end), [*members, finding])
        else:
            groups.append((start, end, [finding]))

    out = text
    for start, end, members in reversed(groups):
        label = min(members, key=lambda f: (-f.score, f.entity_type)).entity_type
        out = out[:start] + placeholder(label) + out[end:]
    return out
