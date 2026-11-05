"""Wrappers that adapt the existing detectors to the pipeline's Detector protocol.

The detectors themselves (input_rules, pii, output_filter) are called
unchanged. The wrappers only translate their outputs into Hits: reason
strings map to fixed entity types and are never stored. Nothing loads at
import; the PII and output-leak models load on the first detect() call.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, field_validator

from llmfirewall.engine import models
from llmfirewall.engine.decision import Span
from llmfirewall.engine.detectors import input_rules, output_filter, pii
from llmfirewall.engine.pipeline import CostClass, DetectorOutputError, Hit
from llmfirewall.engine.policy import DetectorParams, NoParams

# input_rules ---------------------------------------------------------------

INPUT_RULE_TYPES = {
    "Empty query": "EMPTY_INPUT",
    "Query too long": "INPUT_TOO_LONG",
    "CSV injection attempt detected": "CSV_INJECTION",
    "Prompt injection detected": "PROMPT_INJECTION",
    "Suspicious input pattern": "SUSPICIOUS_PATTERN",
}


class InputRulesDetector:
    name: str = "input_rules"
    cost: CostClass = CostClass.CHEAP
    params_model: type[DetectorParams] = NoParams

    def detect(self, text: str, params: DetectorParams) -> list[Hit]:
        # validate_input ignores its role argument (pinned by the
        # characterization tests); "employee" matches those tests.
        valid, reason = input_rules.validate_input(text, "employee")
        if valid:
            return []
        if reason not in INPUT_RULE_TYPES:
            raise DetectorOutputError("unrecognised input_rules reason")
        return [Hit(entity_type=INPUT_RULE_TYPES[reason], score=1.0, span=None)]


# pii -----------------------------------------------------------------------


class PiiParams(DetectorParams):
    entities: tuple[str, ...] = tuple(sorted(pii.REDACT_ENTITIES))
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("entities", mode="before")
    @classmethod
    def _entities_from_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(sorted(set(value)))
        return value

    @field_validator("entities")
    @classmethod
    def _known_entities(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("entities must not be empty")
        unknown = sorted(set(value) - set(pii.REDACT_ENTITIES))
        if unknown:
            raise ValueError(f"unknown entities {unknown}")
        return value

    def stricter(self, other: Self) -> Self:
        """More entities and a lower threshold both find more."""
        return type(self)(
            entities=tuple(sorted(set(self.entities) | set(other.entities))),
            min_score=min(self.min_score, other.min_score),
        )


class PiiDetector:
    name: str = "pii"
    cost: CostClass = CostClass.EXPENSIVE
    params_model: type[DetectorParams] = PiiParams

    def detect(self, text: str, params: DetectorParams) -> list[Hit]:
        if not isinstance(params, PiiParams):
            raise TypeError("pii expects PiiParams")
        # The same analyzer call redact_pii makes; the policy's params then
        # filter its results.
        results = models.presidio_analyzer().analyze(
            text=text, language="en", entities=pii.REDACT_ENTITIES
        )
        return [
            Hit(
                entity_type=r.entity_type,
                score=float(r.score),
                span=Span(start=r.start, end=r.end),
            )
            for r in results
            if r.entity_type in params.entities and r.score >= params.min_score
        ]


# output_leak ---------------------------------------------------------------

OUTPUT_LEAK_PREFIXES = (
    ("Blocked: Pattern detected:", "SECRET_PATTERN"),
    ("Blocked: Similar to", "SECRET_SIMILARITY"),
    ("Blocked: Sensitive keyword:", "SENSITIVE_KEYWORD"),
)


class OutputLeakDetector:
    name: str = "output_leak"
    cost: CostClass = CostClass.EXPENSIVE
    params_model: type[DetectorParams] = NoParams

    def detect(self, text: str, params: DetectorParams) -> list[Hit]:
        passed, reason, score = output_filter.validate_output(text)
        if passed:
            return []
        for prefix, entity_type in OUTPUT_LEAK_PREFIXES:
            if reason.startswith(prefix):
                clamped = min(max(float(score), 0.0), 1.0)
                return [Hit(entity_type=entity_type, score=clamped, span=None)]
        raise DetectorOutputError("unrecognised output_filter reason")
