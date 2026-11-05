"""Fake detectors and policy builders for the engine tests.

The fakes implement the Detector protocol from llmfirewall.engine.pipeline
(name, cost, params_model, detect), so the engine tests do not depend on
model output and run offline.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from pydantic import Field

from llmfirewall.engine.decision import Span
from llmfirewall.engine.pipeline import CostClass, Hit
from llmfirewall.engine.policy import DetectorParams, NoParams, Policy, parse_policy

CHEAP = CostClass.CHEAP
EXPENSIVE = CostClass.EXPENSIVE


class ThresholdParams(DetectorParams):
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)

    def stricter(self, other: ThresholdParams) -> ThresholdParams:
        return ThresholdParams(min_score=min(self.min_score, other.min_score))


class FakeClock:
    """A monotonic clock in seconds that only moves when told to."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance_ms(self, ms: float) -> None:
        self.now += ms / 1000


Finder = Callable[[str], Iterable[Hit]]


def finds(needle: str, entity_type: str, score: float = 1.0) -> Finder:
    """Hits for every occurrence of `needle`, with spans."""

    def find(text: str) -> list[Hit]:
        hits, start = [], text.find(needle)
        while start != -1:
            span = Span(start=start, end=start + len(needle))
            hits.append(Hit(entity_type=entity_type, score=score, span=span))
            start = text.find(needle, start + 1)
        return hits

    return find


def flags(needle: str, entity_type: str, score: float = 1.0) -> Finder:
    """One whole-message hit (no span) when `needle` is in the text."""

    def find(text: str) -> list[Hit]:
        if needle in text:
            return [Hit(entity_type=entity_type, score=score, span=None)]
        return []

    return find


def nothing(text: str) -> list[Hit]:
    return []


class FakeDetector:
    """A detector whose behavior the test chooses.

    call_log, when given, is shared between detectors so a test can check the
    order they ran in. With `clock` and `takes_ms`, each call advances the
    fake clock, to simulate a slow detector without sleeping.
    """

    def __init__(
        self,
        name: str,
        cost: CostClass,
        find: Finder = nothing,
        *,
        error: Exception | None = None,
        clock: FakeClock | None = None,
        takes_ms: float = 0.0,
        params_model: type[DetectorParams] = NoParams,
        call_log: list[str] | None = None,
    ) -> None:
        self.name = name
        self.cost = cost
        self.params_model = params_model
        self._find = find
        self._error = error
        self._clock = clock
        self._takes_ms = takes_ms
        self._call_log = call_log
        self.calls = 0

    def detect(self, text: str, params: DetectorParams) -> Sequence[Hit]:
        self.calls += 1
        if self._call_log is not None:
            self._call_log.append(self.name)
        if self._clock is not None:
            self._clock.advance_ms(self._takes_ms)
        if self._error is not None:
            raise self._error
        return list(self._find(text))


def detector_config(
    action: str = "block",
    apply_to: Sequence[str] = ("user",),
    *,
    enabled: bool = True,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "enabled": enabled,
        "action": action,
        "apply_to": list(apply_to),
    }
    if params is not None:
        config["params"] = params
    return config


def policy_data(
    detectors: dict[str, dict[str, Any]],
    *,
    on_detector_error: str = "fail_closed",
    on_timeout: str = "fail_closed",
    latency_budget_ms: int = 1000,
) -> dict[str, Any]:
    return {
        "version": 1,
        "on_detector_error": on_detector_error,
        "on_timeout": on_timeout,
        "latency_budget_ms": latency_budget_ms,
        "detectors": detectors,
    }


def make_policy(
    detectors: Sequence[FakeDetector],
    configs: dict[str, dict[str, Any]],
    *,
    name: str = "test",
    **options: Any,
) -> Policy:
    return parse_policy(
        policy_data(configs, **options), name=name, detectors=list(detectors)
    )


def user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}
