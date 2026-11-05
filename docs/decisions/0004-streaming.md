# 0004: Check streamed responses with a hold-back window

- Status: accepted (the strategy). N, C and S are settings; the final N is chosen in the latency task.
- Date: 2026-11-19

## Context

Most chat clients stream. The model's answer arrives as many small chunks (in the mock, 3 characters each), and the gateway forwards each chunk as soon as it arrives. The firewall must check the answer before the client sees it, but:

- **post_call is too late for streams.** LiteLLM's `async_post_call_success_hook` runs only after the whole stream has been delivered, and the proxy does not call it at all for a guardrail that defines the streaming iterator hook ([ADR 0003](0003-litellm-guardrail.md)). Checking there would only log a leak that has already happened.
- **A secret can be split across chunks.** `sk-abc…` may arrive as `sk-`, `abc`, `def`, … No single chunk matches a pattern, so checking chunk by chunk finds nothing.
- **Checking is not free.** `pii` (Presidio) and `output_leak` (regex plus MiniLM similarity) take milliseconds per call (warm p95 4.5 ms and 18 ms in docs/baseline.md), and the engine runs on one worker thread. Checking the whole answer so far on every chunk would be quadratic.

The hook used is `async_post_call_streaming_iterator_hook`, which wraps the whole chunk iterator. The adapter decides which text to forward, and when.

## Options

1. **Buffer the whole response, check once, then send it.**
   - Pro: exactly the non-streaming decision; no window effects; simplest to reason about.
   - Con: time to first token (TTFT) becomes the full generation time, so streaming no longer helps the user. A 1,000-character answer at 200 chars/s waits about 5 s.
2. **Hold back a window of N characters** (chosen). Scan the new text together with C characters already sent, and release only text that has been scanned and is more than N characters from the end.
   - Pro: TTFT grows by about the time to generate N characters plus one scan; the rest of the stream flows at the model's pace.
   - Con: N has to be at least as long as the longest text a detector needs before it fires, otherwise a prefix of a match can be released (shown by a test). Detectors that judge the whole text (semantic similarity, NER) see windows, not the whole answer, until the final check.
3. **Release at sentence or line boundaries.**
   - Pro: natural for NER and similarity.
   - Con: no bound on the hold: one long line (a code block, a key with no spaces) holds everything, and the guarantee depends on the model's punctuation.
4. **Check each chunk alone.**
   - Con: misses every match split across chunks. Rejected.
5. **post_call after the stream.**
   - Con: runs after delivery. Rejected.

## Decision

Option 2, implemented in `src/llmfirewall/engine/stream.py` (`StreamWindow`) and used by the adapter's streaming hook. One window per choice index.

**Rules**
1. The last **N** characters received are never sent (`LLMFW_STREAM_HOLDBACK_CHARS`, default 128).
2. Once more than N characters are pending and at least **S** new characters have arrived since the last scan (`LLMFW_STREAM_SCAN_STEP_CHARS`, default 32), `Engine.check_response` runs on the last **C** characters already sent (`LLMFW_STREAM_CONTEXT_CHARS`, default 128) plus everything pending.
3. If the decision is **block**, the stream stops. The client gets one final chunk with no content and `finish_reason: "content_filter"`, and nothing still held back is ever sent. There is no SSE error event. The audit event records the stop: `stream_stop = {reason: "content_filter", choice_index, released_chars, withheld_chars}`.
4. Otherwise, everything except the last N characters is released, with redaction placeholders applied. A redaction span that crosses the release point holds back from where the span starts, so a match is never split and its tail never sent. An open-ended match (a long key) keeps being held until it ends.
5. At the end of the stream, `check_response` runs once on the **whole** response, the same check a non-streaming response gets. Its decision releases, redacts or withholds what is still held back.
6. Context text is kept unredacted in memory, so patterns see the real text. The whole response is kept in memory for the final check. Neither is ever logged.
7. Any exception, an unknown chunk type or non-text content ends the stream with `content_filter`, which is failing closed (ADR 0002).
8. Tool-call deltas pass through unchanged (tool-call arguments are v2).

**What N must be for the formats we detect.** For a match to be caught before any character of it is released, N must be at least the length of the shortest text that makes its detector fire. In the worst case, the match completes with a 1-character chunk right after a release.

| Format | Detector | Pattern (abridged) | Characters before it fires |
|---|---|---|---|
| Local phone | pii | `\d{3}[-.\s]\d{4}` | 8 |
| SSN | pii, output_leak | `\d{3}[-\s]?\d{2}[-\s]?\d{4}` | up to 11 |
| Phone | output_leak | `\d{3}[-.\s]\d{3}[-.\s]\d{4}` | 12 |
| IPv4 / IPv6 | pii | custom / Presidio built-in | up to 15 / up to 45 |
| Credit card | pii | 4×4 digits with separators; Presidio built-in | up to 19 / about 23 |
| Numeric code | output_leak | `(passcode\|password\|code\|pin)[\s:is]*\d{4,6}` | about 20 (`passcode is: 123456`) |
| Password | output_leak | `password\s*[:=]\s*\S+` | 10, then open-ended |
| `sk-` key | pii / output_leak | `sk-[a-zA-Z0-9]{15,}` / `{20,}` | 18 / 23, then open-ended |
| 32+ letters/digits | output_leak | `[A-Z0-9]{32,}` (case-insensitive) | 32, then open-ended |
| Salary | output_leak | `\$\d{1,3}(,\d{3})*` | 2, then open-ended |
| Email | pii, output_leak | `local@domain.tld` | the whole local part (up to 64 by RFC 5321) plus `@` and the domain |
| Budget | output_leak | `budget.*\$\d+` | open gap within a line |
| PERSON, LOCATION | pii (spaCy NER) | not length-defined | depends on context |

So:
- **N ≥ 45** covers every fixed-length format.
- **N ≥ 32** covers the trigger length of every open-ended format. Rule 4 then holds the tail until the match ends.
- **N = 128** also covers an email with a 64-character local part and a domain of up to 63 characters.
- **C** must be at least the prefix a pattern needs from already-sent text (about 16 for `passcode is: `). C = 128 also covers `budget … $` gaps of up to 128 characters.

**Known limits of the window.**
- `budget.*\$\d+` with more than C characters between "budget" and the amount is caught only by the final check, and only if the amount is still held back.
- The semantic-similarity part of `output_leak` and spaCy's PERSON/LOCATION recognition see windows of at most C + N + S characters, not the whole answer. They may fire differently than on the full text. The final full-text check catches what only shows up in the whole answer, but by then only the last N characters can still be withheld.
- Redactions found only by the final check in text already sent cannot be undone. They are recorded in the audit event.
- Answers shorter than N characters arrive all at once at the end, exactly as in option 1.
- The engine runs on one worker thread, so concurrent streams queue for scans.

## Measured time-to-first-token cost

`bench/stream_ttft.py`, run on 2026-11-19 on an Apple M3, Python 3.12.16, CPU torch, Hugging Face offline:
- **Setup:** the adapter's streaming hook runs in-process with the real detectors and the shipped `default` policy, with C=128 and S=32. It is fed by a mocked upstream that yields LiteLLM `ModelResponseStream` chunks of 3 characters at a fixed rate.
- **Response:** a benign 1,054-character answer with nothing to redact or block (the common case).
- **Runs:** 30 per configuration, after warm-up.
- **TTFT** is the time from the start of the stream to the first chunk with content. "baseline" reads the same upstream without the hook.

| Upstream rate | Config | TTFT p50 (ms) | TTFT p95 (ms) | Whole stream p50 (ms) | Whole stream p95 (ms) |
|---|---|---|---|---|---|
| 200 chars/s (~50 tok/s) | baseline | 16.3 | 17.4 | 5690.1 | 5740.9 |
| 200 chars/s | N=64 | 389.7 | 404.2 | 6723.2 | 6843.1 |
| 200 chars/s | N=128 | 739.0 | 755.3 | 6718.1 | 6796.4 |
| 200 chars/s | N=256 | 1455.3 | 1473.1 | 6681.6 | 6820.3 |
| 800 chars/s (~200 tok/s) | baseline | 4.4 | 4.5 | 1540.8 | 1575.6 |
| 800 chars/s | N=64 | 112.1 | 114.6 | 2123.3 | 2158.4 |
| 800 chars/s | N=128 | 215.9 | 223.9 | 2124.3 | 2153.6 |
| 800 chars/s | N=256 | 424.1 | 436.8 | 2108.7 | 2168.8 |

Reading the numbers:
- **TTFT ≈ N / rate + the first scan.** Each doubling of N adds about N/rate: at 200 chars/s, 64→128 adds 349 ms (64/200 = 320 ms) and 128→256 adds 716 ms (640 ms). What is left over is the first scan of a window of about N characters: roughly 50–175 ms at 200 chars/s, growing with N because Presidio and MiniLM cost more on longer text. With the default N=128, the hook adds **about 0.72 s to TTFT at 200 chars/s and 0.21 s at 800 chars/s**.
- **The whole stream takes 0.6–1.0 s longer, whatever N is.** That is the ~33 scans (one per 32 characters) plus the final full-text check. Treat this as an upper bound: the benchmark's mocked upstream produces the next chunk only when asked for it, so scan time adds up serially. A real upstream keeps generating into the socket buffer while a scan runs. It was not measured with concurrent streams (the engine has one worker thread).
- Numbers from one machine. The latency task will set the budget and enforce it in CI.

## Consequences

- Streaming keeps working with a bounded delay, and a secret split across chunks never reaches the client. This is proven by tests at three levels (Evidence).
- A blocked stream looks like a normal end of stream with `finish_reason: "content_filter"`. Clients that ignore `finish_reason` show the partial answer without an error.
- N, C and S are settings, so the latency task can choose N against the TTFT budget. Lowering N below the trigger lengths above reopens prefix leaks; a unit test shows this.
- Follow-up: a CI latency gate on TTFT, and a concurrency benchmark (the single engine thread).

## Evidence

- `tests/unit/test_stream_window.py`:
  - a secret split at every possible point, and in 200 random chunkings, never reaches the client, and the output equals non-streaming redaction;
  - an open-ended key longer than N is redacted whole;
  - blocks withhold the held text;
  - the final full-text check is applied.
- `tests/unit/test_litellm_guardrail.py`:
  - the hook ends a blocked stream with a `content_filter` chunk and no content;
  - the audit `stream_stop` is recorded;
  - `test_holdback_shorter_than_the_trigger_length_leaks_a_prefix` shows why N matters.
- `tests/integration/test_litellm_adapter.py`: real detectors, with LiteLLM's own `CustomStreamWrapper` streaming a key in 3-character chunks (three chunk alignments). The key is redacted (redact-only policy) or blocked (`content_filter`, shipped default), and no 4 consecutive characters of it reach the client.
- `tests/e2e/test_proxy.py`: through `docker compose up`, with the openai SDK and the raw SSE bytes on the wire. The mock upstream streams a made-up key in 3-character chunks; the client receives no part of it, and the stream ends with `content_filter` and no SSE error.
- TTFT: `bench/stream_ttft.py`, numbers above.
