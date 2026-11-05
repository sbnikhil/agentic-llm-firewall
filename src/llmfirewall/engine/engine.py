"""Engine.check_request / Engine.check_response.

Design: docs/design/policy.md section 7. The engine receives a Policy that
the caller resolved from verified identity (identity.py); it never reads a
policy name from the messages.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from llmfirewall.engine.decision import (
    Action,
    Decision,
    DetectorFailure,
    Finding,
    redact_text,
)
from llmfirewall.engine.pipeline import (
    Clock,
    Detector,
    PipelineResult,
    Segment,
    run_pipeline,
)
from llmfirewall.engine.policy import REQUEST_ROLES, Policy, PolicyError, Role


def builtin_detectors() -> list[Detector]:
    """The wrappers for the existing detectors. Loads no model."""
    from llmfirewall.engine.detectors.builtin import (
        InputRulesDetector,
        OutputLeakDetector,
        PiiDetector,
    )

    return [InputRulesDetector(), PiiDetector(), OutputLeakDetector()]


class UnsupportedInput(Exception):
    def __init__(self, message_index: int) -> None:
        super().__init__(f"message {message_index} cannot be checked")
        self.message_index = message_index


def _message_text(message: Any, index: int) -> tuple[Role, str | None]:
    """The role and the checkable text of one message; raises
    UnsupportedInput for anything the engine cannot check."""
    if not isinstance(message, Mapping):
        raise UnsupportedInput(index)
    role = message.get("role")
    if not isinstance(role, str) or role not in REQUEST_ROLES:
        raise UnsupportedInput(index)
    content = message.get("content")
    if content is None:
        return Role(role), None
    if isinstance(content, str):
        return Role(role), content
    if isinstance(content, list):
        texts = []
        for part in content:
            if (
                not isinstance(part, Mapping)
                or part.get("type") != "text"
                or not isinstance(part.get("text"), str)
            ):
                raise UnsupportedInput(index)
            texts.append(part["text"])
        return Role(role), "\n".join(texts)
    raise UnsupportedInput(index)


class Engine:
    def __init__(
        self,
        detectors: Sequence[Detector] | None = None,
        *,
        clock: Clock = time.monotonic,
    ) -> None:
        detectors = builtin_detectors() if detectors is None else list(detectors)
        self._detectors: dict[str, Detector] = {}
        for detector in detectors:
            if detector.name in self._detectors:
                raise PolicyError(f"two detectors are named {detector.name!r}")
            self._detectors[detector.name] = detector
        self._clock = clock

    def _check_policy(self, policy: Policy) -> None:
        if set(policy.detectors) != set(self._detectors):
            raise PolicyError(
                f"policy {policy.name!r} was not parsed for this engine's detectors"
            )
        for name, config in policy.detectors.items():
            if type(config.params) is not self._detectors[name].params_model:
                raise PolicyError(
                    f"policy {policy.name!r}: params for {name!r} do not match "
                    "the engine's detector"
                )

    def _unsupported(self, policy: Policy, message_index: int) -> Decision:
        return Decision(
            action=Action.BLOCK,
            policy_name=policy.name,
            findings=(),
            failures=(
                DetectorFailure(
                    detector="engine",
                    kind="unsupported_input",
                    error_type=None,
                    message_index=message_index,
                ),
            ),
            skipped=tuple(sorted(n for n, c in policy.detectors.items() if c.enabled)),
        )

    def _run(self, segments: list[Segment], policy: Policy) -> PipelineResult:
        return run_pipeline(segments, policy, self._detectors, self._clock)

    def _redact_findings(self, result: PipelineResult, policy: Policy) -> list[Finding]:
        return [
            f
            for f in result.findings
            if policy.detectors[f.detector].action is Action.REDACT
            and f.span is not None
        ]

    def check_request(
        self, messages: Sequence[Mapping[str, Any]], policy: Policy
    ) -> Decision:
        self._check_policy(policy)
        segments: list[Segment] = []
        for index, message in enumerate(messages):
            try:
                role, text = _message_text(message, index)
            except UnsupportedInput as e:
                return self._unsupported(policy, e.message_index)
            if text is not None:
                segments.append(Segment(role=role, message_index=index, text=text))

        result = self._run(segments, policy)
        redacted: tuple[dict[str, Any], ...] | None = None
        if result.action is Action.REDACT:
            to_redact = self._redact_findings(result, policy)
            texts = {s.message_index: s.text for s in segments}
            copies = [dict(m) for m in messages]
            indices = {f.message_index for f in to_redact}
            for index in sorted(i for i in indices if i is not None):
                copies[index]["content"] = redact_text(
                    texts[index], [f for f in to_redact if f.message_index == index]
                )
            redacted = tuple(copies)
        return self._decision(result, policy, redacted_messages=redacted)

    def check_response(self, text: str, policy: Policy) -> Decision:
        self._check_policy(policy)
        segment = Segment(role=Role.RESPONSE, message_index=None, text=text)
        result = self._run([segment], policy)
        redacted: str | None = None
        if result.action is Action.REDACT:
            redacted = redact_text(text, self._redact_findings(result, policy))
        return self._decision(result, policy, redacted_text=redacted)

    @staticmethod
    def _decision(
        result: PipelineResult,
        policy: Policy,
        *,
        redacted_messages: tuple[dict[str, Any], ...] | None = None,
        redacted_text: str | None = None,
    ) -> Decision:
        return Decision(
            action=result.action,
            policy_name=policy.name,
            findings=tuple(result.findings),
            failures=tuple(result.failures),
            skipped=tuple(result.skipped),
            durations_ms=dict(result.durations_ms),
            redacted_messages=redacted_messages,
            redacted_text=redacted_text,
        )
