"""Check a streamed response without sending text that has not been checked.

Strategy (docs/decisions/0004-streaming.md):
- The last N characters received (`holdback`) are never sent.
- Once more than N characters are pending and at least S new characters
  (`step`) arrived since the last scan, the pending text is scanned together
  with the last C characters already sent (`context`), and everything except
  the last N characters is released, with redactions applied.
- A redaction span that crosses the release point holds back from where the
  span starts, so a long match is never split and its tail never sent.
- A block stops the stream: nothing still held back is ever sent.
- At the end, the whole response is checked once (the same check a
  non-streaming response gets), and the held-back text is released or
  withheld by that decision.

Everything here is synchronous; the adapter runs it off the event loop.
The full text is kept in memory for the final check and is never logged.
"""

from __future__ import annotations

from dataclasses import dataclass

from llmfirewall.engine.audit import AuditFinding, findings_of
from llmfirewall.engine.decision import Action, Decision, Finding, Span, redact_text
from llmfirewall.engine.engine import Engine
from llmfirewall.engine.policy import Policy
from llmfirewall.engine.settings import Settings


@dataclass(frozen=True)
class WindowConfig:
    holdback: int  # N
    context: int  # C
    step: int  # S

    def __post_init__(self) -> None:
        if self.holdback < 0 or self.context < 0 or self.step < 1:
            raise ValueError("holdback and context must be >= 0, step >= 1")

    @classmethod
    def from_settings(cls, settings: Settings) -> WindowConfig:
        return cls(
            holdback=settings.stream_holdback_chars,
            context=settings.stream_context_chars,
            step=settings.stream_scan_step_chars,
        )


@dataclass(frozen=True)
class Release:
    text: str  # checked (and redacted) text to send now; may be empty
    stopped: bool  # True: blocked; send nothing more


@dataclass(frozen=True)
class _Span:
    start: int  # offsets in the whole response
    end: int
    finding: Finding


class StreamWindow:
    """One choice of one streamed response."""

    def __init__(self, engine: Engine, policy: Policy, config: WindowConfig) -> None:
        self._engine = engine
        self._policy = policy
        self._config = config
        self._text = ""
        self._released = 0  # original characters already sent
        self._unscanned = 0  # characters received since the last scan
        self.stopped = False
        self.finished = False
        self.decisions: list[Decision] = []
        self.findings: list[AuditFinding] = []

    @property
    def received_chars(self) -> int:
        return len(self._text)

    @property
    def released_chars(self) -> int:
        return self._released

    @property
    def withheld_chars(self) -> int:
        return len(self._text) - self._released

    def feed(self, delta: str) -> Release:
        if self.stopped or self.finished:
            raise RuntimeError("feed() after the window stopped or finished")
        self._text += delta
        self._unscanned += len(delta)
        pending = len(self._text) - self._released
        if pending <= self._config.holdback or self._unscanned < self._config.step:
            return Release("", False)
        context_start = max(0, self._released - self._config.context)
        cut = len(self._text) - self._config.holdback
        return self._check(context_start, cut)

    def finish(self) -> Release:
        """End of the stream: check the whole response, release the rest."""
        if self.stopped or self.finished:
            raise RuntimeError("finish() after the window stopped or finished")
        self.finished = True
        if not self._text:
            return Release("", False)
        return self._check(0, len(self._text))

    def _check(self, window_start: int, cut: int) -> Release:
        decision = self._engine.check_response(self._text[window_start:], self._policy)
        self.decisions.append(decision)
        self.findings.extend(findings_of(decision, offset=window_start))
        self._unscanned = 0
        if decision.action is Action.BLOCK:
            self.stopped = True
            return Release("", True)

        spans = [
            _Span(f.span.start + window_start, f.span.end + window_start, f)
            for f in decision.findings
            if f.span is not None
            and self._policy.detectors[f.detector].action is Action.REDACT
        ]
        cut = _pull_back(cut, spans)
        cut = max(cut, self._released)
        text = self._redact(self._released, cut, spans)
        self._released = cut
        return Release(text, False)

    def _redact(self, start: int, end: int, spans: list[_Span]) -> str:
        if end <= start:
            return ""
        segment = self._text[start:end]
        # Spans are clipped to the segment. A span that began in text already
        # sent has its remaining part redacted here.
        clipped = [
            s.finding.model_copy(
                update={
                    "span": Span(
                        start=max(s.start, start) - start,
                        end=min(s.end, end) - start,
                    ),
                    "message_index": None,
                }
            )
            for s in spans
            if s.end > start and s.start < end
        ]
        return redact_text(segment, clipped) if clipped else segment


def _pull_back(cut: int, spans: list[_Span]) -> int:
    """Move the release point left until no span crosses it."""
    moved = True
    while moved:
        moved = False
        for s in spans:
            if s.start < cut < s.end:
                cut = s.start
                moved = True
    return cut
