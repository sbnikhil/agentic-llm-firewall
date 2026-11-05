"""engine.stream.StreamWindow: no unchecked text is ever released.

Fake detectors stand in for the real ones; the property tests split each
response at every possible point (and into many random chunkings) and check
what the client would have received.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable

import pytest

from llmfirewall.engine.decision import Span
from llmfirewall.engine.engine import Engine
from llmfirewall.engine.pipeline import Hit
from llmfirewall.engine.stream import Release, StreamWindow, WindowConfig
from tests.fake_secrets import STREAM_API_KEY
from tests.unit.engine_fakes import (
    EXPENSIVE,
    FakeDetector,
    detector_config,
    flags,
    make_policy,
)

SECRET_RE = re.compile(r"sk-[A-Za-z0-9]{8,}")


def regex_finder(pattern: re.Pattern[str], entity_type: str):
    def find(text: str) -> list[Hit]:
        return [
            Hit(
                entity_type=entity_type,
                score=1.0,
                span=Span(start=m.start(), end=m.end()),
            )
            for m in pattern.finditer(text)
        ]

    return find


def make(
    holdback: int = 16,
    context: int = 16,
    step: int = 1,
    *,
    block_on: str = "BOOM",
) -> tuple[Engine, StreamWindow, object]:
    secret = FakeDetector("secret", EXPENSIVE, regex_finder(SECRET_RE, "API_KEY"))
    bomb = FakeDetector("bomb", EXPENSIVE, flags(block_on, "BOMB"))
    detectors = [secret, bomb]
    policy = make_policy(
        detectors,
        {
            "secret": detector_config("redact", ["response"]),
            "bomb": detector_config("block", ["response"]),
        },
    )
    engine = Engine(detectors)
    window = StreamWindow(engine, policy, WindowConfig(holdback, context, step))
    return engine, window, policy


def run(window: StreamWindow, chunks: Iterable[str]) -> tuple[str, bool]:
    """What the client receives, and whether the stream was stopped."""
    out = []
    for chunk in chunks:
        r = window.feed(chunk)
        out.append(r.text)
        if r.stopped:
            return "".join(out), True
    r = window.finish()
    out.append(r.text)
    return "".join(out), r.stopped


def splits(text: str, sizes: Iterable[int]) -> list[str]:
    chunks, i = [], 0
    for size in sizes:
        if i >= len(text):
            break
        chunks.append(text[i : i + size])
        i += size
    if i < len(text):
        chunks.append(text[i:])
    return chunks


def random_chunkings(text: str, n: int, seed: int = 7) -> list[list[str]]:
    rng = random.Random(seed)  # noqa: S311 (test chunking)
    return [
        splits(text, (rng.randint(1, 9) for _ in range(len(text)))) for _ in range(n)
    ]


TEXT = f"Here is the key you asked for: {STREAM_API_KEY} and some more words after it."
EXPECTED = (
    "Here is the key you asked for: [REDACTED:API_KEY] and some more words after it."
)


def test_unsplit_stream_matches_non_streaming_redaction() -> None:
    _, window, _ = make()
    assert run(window, [TEXT]) == (EXPECTED, False)


@pytest.mark.parametrize("cut", range(1, len(TEXT)))
def test_secret_split_at_any_point_never_reaches_the_client(cut: int) -> None:
    _, window, _ = make()
    received, stopped = run(window, [TEXT[:cut], TEXT[cut:]])
    assert not stopped
    assert received == EXPECTED
    assert "sk-" not in received


@pytest.mark.parametrize("chunks", random_chunkings(TEXT, 200))
def test_secret_in_random_chunks_never_reaches_the_client(chunks: list[str]) -> None:
    _, window, _ = make(holdback=16, context=16, step=4)
    received, stopped = run(window, chunks)
    assert received == EXPECTED


def test_every_released_prefix_contains_no_part_of_the_secret() -> None:
    """Character by character: at no point has the client seen any of it."""
    _, window, _ = make()
    secret = STREAM_API_KEY
    start = TEXT.index(secret)
    received = ""
    for ch in TEXT:
        received += window.feed(ch).text
        # Whatever was released so far is an exact prefix of the redacted
        # text, so it can only contain the placeholder, never key characters.
        assert EXPECTED.startswith(received)
        if len(received) > start:
            assert received[start:].startswith(
                "[REDACTED:API_KEY]"[: len(received) - start]
            )


def test_secret_longer_than_the_holdback_is_redacted_whole() -> None:
    """An open-ended match keeps holding until it ends: no tail leaks."""
    long_key = "sk-" + "A" * 100
    text = f"key {long_key} done and then plenty of trailing text here"
    _, window, _ = make(holdback=16, context=16, step=1)
    received, _ = run(window, list(text))
    assert (
        received == "key [REDACTED:API_KEY] done and then plenty of trailing text here"
    )
    assert "AAAA" not in received


def test_nothing_is_released_until_more_than_holdback_characters_arrive() -> None:
    _, window, _ = make(holdback=16, step=1)
    for ch in "x" * 16:
        assert window.feed(ch) == Release("", False)
    assert window.feed("y").text == "x"


def test_scan_waits_for_the_step() -> None:
    engine, window, _ = make(holdback=4, step=8)
    for ch in "abcdefg":
        window.feed(ch)
    assert window.decisions == []
    window.feed("h")
    assert len(window.decisions) == 1


def test_block_stops_the_stream_and_withholds_the_held_text() -> None:
    text = "A perfectly normal sentence that goes on for a while. BOOM and the rest."
    _, window, _ = make(holdback=16, step=1)
    received, stopped = run(window, list(text))
    assert stopped
    assert "BOOM" not in received
    assert "BOO" not in received
    assert window.withheld_chars > 0
    assert window.released_chars + window.withheld_chars == window.received_chars


@pytest.mark.parametrize("cut", range(1, 30))
def test_block_split_across_chunks_withholds_everything_after_release(cut: int) -> None:
    text = "intro text is here BOOM then"
    _, window, _ = make(holdback=8, step=1)
    received, stopped = run(window, [text[:cut], text[cut:]])
    assert stopped
    assert "B" not in received[text.index("BOOM") :]
    assert "BOOM" not in received


def test_feed_after_stop_is_an_error() -> None:
    _, window, _ = make(holdback=0, step=1)
    assert window.feed("BOOM").stopped
    with pytest.raises(RuntimeError):
        window.feed("more")
    with pytest.raises(RuntimeError):
        window.finish()


def test_final_check_runs_on_the_whole_response() -> None:
    """A finding that only exists in the full text still withholds the
    held-back tail."""
    whole = FakeDetector(
        "secret",
        EXPENSIVE,
        lambda text: (
            [Hit(entity_type="LONG", score=1.0, span=None)] if len(text) > 60 else []
        ),
    )
    bomb = FakeDetector("bomb", EXPENSIVE)
    policy = make_policy(
        [whole, bomb],
        {
            "secret": detector_config("block", ["response"]),
            "bomb": detector_config("block", ["response"]),
        },
    )
    window = StreamWindow(Engine([whole, bomb]), policy, WindowConfig(10, 10, 1))
    text = "y" * 61
    received, stopped = run(window, list(text))
    assert stopped
    assert len(received) < len(text)
    assert len(window.decisions[-1].findings) == 1


def test_context_lets_a_match_that_began_in_sent_text_block() -> None:
    """'BO' was sent as context; 'OM' arrives later; the window sees 'BOOM'."""
    _, window, _ = make(holdback=2, context=8, step=1, block_on="xxBOOM")
    received, stopped = run(window, list("....xxBOOM tail"))
    assert stopped


def test_findings_are_recorded_in_whole_response_offsets() -> None:
    _, window, _ = make(holdback=8, context=8, step=1)
    run(window, list(TEXT))
    start = TEXT.index("sk-")
    spans = {(f.start, f.end) for f in window.findings if f.entity_type == "API_KEY"}
    assert (start, start + len(STREAM_API_KEY)) in spans


def test_empty_stream() -> None:
    _, window, _ = make()
    assert run(window, []) == ("", False)
    assert window.decisions == []


@pytest.mark.parametrize(
    ("holdback", "context", "step"), [(-1, 0, 1), (0, -1, 1), (0, 0, 0)]
)
def test_invalid_config(holdback: int, context: int, step: int) -> None:
    with pytest.raises(ValueError):
        WindowConfig(holdback, context, step)
