"""Run a policy's detectors over text segments, cheapest first.

Design: docs/design/policy.md section 6.
- Detectors run ordered by cost class, then by name.
- Once the running action is BLOCK, the rest of the current cost class still
  runs (its findings are recorded) and every later class is skipped.
- The latency budget is checked on the clock after each detector call; there
  are no threads, so a single slow call is not interrupted. When the budget
  is exceeded, the remaining calls are skipped and on_timeout applies.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from llmfirewall.engine.decision import (
    Action,
    DetectorFailure,
    FailureKind,
    Finding,
    Span,
    Verdict,
)
from llmfirewall.engine.policy import DetectorParams, FailureMode, Policy, Role

Clock = Callable[[], float]  # seconds, monotonic


class CostClass(IntEnum):
    CHEAP = 0  # pure Python, microseconds
    EXPENSIVE = 1  # model inference


class Hit(BaseModel):
    """What a detector reports for one text: no text, only type and place."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entity_type: str
    score: float = Field(ge=0.0, le=1.0)
    span: Span | None


class Detector(Protocol):
    name: str
    cost: CostClass
    params_model: type[DetectorParams]

    def detect(self, text: str, params: DetectorParams) -> Sequence[Hit]: ...


class DetectorOutputError(Exception):
    """A detector produced output the wrapper cannot interpret."""


@dataclass(frozen=True)
class Segment:
    role: Role
    message_index: int | None
    text: str = field(repr=False)


@dataclass
class PipelineResult:
    action: Action
    findings: list[Finding]
    failures: list[DetectorFailure]
    skipped: list[str]
    # Total time per detector across all segments, including detectors
    # that found nothing. For the audit log only; no decision reads it.
    durations_ms: dict[str, float] = field(default_factory=dict)


def _valid_hits(hits: object, text: str) -> list[Hit]:
    if not isinstance(hits, Sequence) or isinstance(hits, str | bytes):
        raise DetectorOutputError("detector must return a sequence of Hit")
    for hit in hits:
        if not isinstance(hit, Hit):
            raise DetectorOutputError("detector must return a sequence of Hit")
        if hit.span is not None and hit.span.end > len(text):
            raise DetectorOutputError("span is outside the text")
    return list(hits)


def run_pipeline(
    segments: Sequence[Segment],
    policy: Policy,
    detectors: Mapping[str, Detector],
    clock: Clock,
) -> PipelineResult:
    verdict = Verdict()
    findings: list[Finding] = []
    failures: list[DetectorFailure] = []
    skipped: list[str] = []
    durations_ms: dict[str, float] = {}
    fail_closed_on_error = policy.on_detector_error is FailureMode.FAIL_CLOSED
    fail_closed_on_timeout = policy.on_timeout is FailureMode.FAIL_CLOSED

    deadline = clock() + policy.latency_budget_ms / 1000
    blocked_in: CostClass | None = None
    timed_out = False

    enabled = [d for d in detectors.values() if policy.detectors[d.name].enabled]
    for detector in sorted(enabled, key=lambda d: (d.cost, d.name)):
        if timed_out or (blocked_in is not None and detector.cost > blocked_in):
            skipped.append(detector.name)
            continue

        config = policy.detectors[detector.name]
        for segment in segments:
            if segment.role not in config.apply_to:
                continue

            started = clock()
            failure: FailureKind | None = None
            error_type: str | None = None
            hits: list[Hit] = []
            try:
                hits = _valid_hits(
                    detector.detect(segment.text, config.params), segment.text
                )
            except DetectorOutputError:
                failure = "invalid_output"
            except Exception as e:  # any detector bug is handled by policy
                failure, error_type = "error", type(e).__name__
            finished = clock()
            durations_ms[detector.name] = durations_ms.get(detector.name, 0.0) + max(
                0.0, (finished - started) * 1000
            )

            if failure is not None:
                failures.append(
                    DetectorFailure(
                        detector=detector.name,
                        kind=failure,
                        error_type=error_type,
                        message_index=segment.message_index,
                    )
                )
                if fail_closed_on_error:
                    verdict.escalate(Action.BLOCK)
            else:
                duration_ms = max(0.0, (finished - started) * 1000)
                for hit in hits:
                    findings.append(
                        Finding(
                            detector=detector.name,
                            entity_type=hit.entity_type,
                            score=hit.score,
                            message_index=segment.message_index,
                            span=hit.span,
                            duration_ms=duration_ms,
                        )
                    )
                    action = config.action
                    if action is Action.REDACT and hit.span is None:
                        action = Action.BLOCK  # nothing to replace
                    verdict.escalate(action)

            if verdict.action is Action.BLOCK and blocked_in is None:
                blocked_in = detector.cost

            if finished > deadline:
                failures.append(
                    DetectorFailure(
                        detector=detector.name,
                        kind="timeout",
                        error_type=None,
                        message_index=segment.message_index,
                    )
                )
                if fail_closed_on_timeout:
                    verdict.escalate(Action.BLOCK)
                timed_out = True
                break

    return PipelineResult(verdict.action, findings, failures, skipped, durations_ms)
