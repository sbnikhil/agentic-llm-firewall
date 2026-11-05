"""Time to first token (TTFT) added by the streaming window (ADR 0004).

The adapter's streaming hook (LLMFirewallGuardrail.
async_post_call_streaming_iterator_hook) runs in-process with the real
detectors and the shipped default policy, fed by a mocked upstream: an async
generator yielding LiteLLM ModelResponseStream chunks of CHUNK_CHARS
characters, one every CHUNK_DELAY ms, so the upstream produces text at a
fixed rate. No proxy, no network; what is measured is only what the
firewall adds.

For each configuration and each run:
  ttft   - time from the start of the stream to the first chunk with
           non-empty content at the client;
  total  - time from the start to the end of the stream.
"baseline" is the same upstream read directly, without the hook.

Configurations: N (hold-back) in 64, 128, 256 with C=128, S=32, at two
upstream rates:
  200 chars/s  (about 50 tokens/s at ~4 chars/token)
  800 chars/s  (about 200 tokens/s)

The response is a benign 1,054-character answer with no PII or secrets
(the common case: nothing is redacted or blocked).

Models load and run once (warm-up) before timing. Torch is forced onto CPU.

Usage (from repo root):
    HF_HUB_OFFLINE=1 LITELLM_LOCAL_MODEL_COST_MAP=True \\
        .venv/bin/python bench/stream_ttft.py [--runs 30]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sentence_transformers  # noqa: E402

_orig_init = sentence_transformers.SentenceTransformer.__init__


def _cpu_init(self: Any, *args: Any, **kwargs: Any) -> None:
    kwargs["device"] = "cpu"
    _orig_init(self, *args, **kwargs)


sentence_transformers.SentenceTransformer.__init__ = _cpu_init  # type: ignore[method-assign]

from litellm.proxy._types import UserAPIKeyAuth  # noqa: E402
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices  # noqa: E402
from pydantic import SecretStr  # noqa: E402

from llmfirewall.adapters.litellm_guardrail import LLMFirewallGuardrail  # noqa: E402
from llmfirewall.engine.settings import REPO_ROOT, Settings  # noqa: E402

CHUNK_CHARS = 3
ANSWER = (
    "Sure. A good way to approach this is to break the problem into smaller "
    "steps and check each one before moving on. First, write down what the "
    "program should do in plain words, and list the inputs it receives and "
    "the outputs it should produce. Second, pick the simplest data structure "
    "that can hold those inputs; a list or a dictionary is usually enough to "
    "start with. Third, write a small function for each step, and test each "
    "function on its own with a couple of examples, including an empty input "
    "and a very large one. Fourth, connect the functions together and run the "
    "whole program on the examples again. If something goes wrong, print the "
    "intermediate values to see where the result first differs from what you "
    "expected. Finally, once it works, read the code again from the top and "
    "remove anything that is no longer needed, rename variables whose names "
    "are unclear, and add a short comment where the reason for a choice is "
    "not obvious from the code itself. This keeps the program easy to change "
    "later, which matters more than making it clever today. "
)
assert 1000 < len(ANSWER) < 1100, len(ANSWER)  # noqa: S101


async def upstream(delay_s: float):
    yield ModelResponseStream(
        id="bench",
        model="mock",
        choices=[StreamingChoices(index=0, delta=Delta(role="assistant", content=""))],
    )
    for i in range(0, len(ANSWER), CHUNK_CHARS):
        await asyncio.sleep(delay_s)
        yield ModelResponseStream(
            id="bench",
            model="mock",
            choices=[
                StreamingChoices(
                    index=0, delta=Delta(content=ANSWER[i : i + CHUNK_CHARS])
                )
            ],
        )
    yield ModelResponseStream(
        id="bench",
        model="mock",
        choices=[
            StreamingChoices(index=0, delta=Delta(content=None), finish_reason="stop")
        ],
    )


async def measure(stream: Any) -> tuple[float, float, str]:
    start = time.perf_counter()
    ttft: float | None = None
    text = ""
    async for chunk in stream:
        for choice in chunk.choices:
            content = choice.delta.content or ""
            if content and ttft is None:
                ttft = time.perf_counter() - start
            text += content
    total = time.perf_counter() - start
    return (ttft if ttft is not None else total), total, text


_IDENTITY = Path(tempfile.mkdtemp(prefix="llmfw-bench-")) / "identity.yaml"
_IDENTITY.write_text("version: 1\ndefault: default\n")


def guard_for(holdback: int) -> LLMFirewallGuardrail:
    return LLMFirewallGuardrail(
        settings=Settings(
            audit_hmac_key=SecretStr("bench"),
            policies_dir=REPO_ROOT / "policies",
            identity_map_path=_IDENTITY,
            stream_holdback_chars=holdback,
            stream_context_chars=128,
            stream_scan_step_chars=32,
        ),
        audit_sink=_NullSink(),
        guardrail_name="llmfirewall",
        default_on=True,
    )


class _NullSink:
    def emit(self, event: Any) -> None:
        pass


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(q * (len(ordered) - 1))))
    return ordered[k]


async def main(runs: int) -> dict[str, Any]:
    auth = UserAPIKeyAuth(
        user_id="bench", team_id=None, allowed_routes=["/v1/chat/completions"]
    )
    guards = {n: guard_for(n) for n in (64, 128, 256)}
    results: dict[str, Any] = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "runs": runs,
        "answer_chars": len(ANSWER),
        "chunk_chars": CHUNK_CHARS,
        "context_chars": 128,
        "step_chars": 32,
        "rows": [],
    }
    for rate in (200, 800):
        delay = CHUNK_CHARS / rate
        holdbacks: list[int | None] = [None, 64, 128, 256]
        configs = [("baseline" if n is None else f"N={n}", n) for n in holdbacks]
        for label, n in configs:
            ttfts, totals = [], []
            for _ in range(runs):
                if n is None:
                    stream = upstream(delay)
                else:
                    stream = guards[n].async_post_call_streaming_iterator_hook(
                        user_api_key_dict=auth,
                        response=upstream(delay),
                        request_data={},
                    )
                ttft, total, text = await measure(stream)
                if text != ANSWER:
                    raise SystemExit(f"{label}: the benign answer was changed")
                ttfts.append(ttft * 1000)
                totals.append(total * 1000)
            results["rows"].append(
                {
                    "rate_chars_per_s": rate,
                    "config": label,
                    "ttft_ms_p50": round(statistics.median(ttfts), 1),
                    "ttft_ms_p95": round(pct(ttfts, 0.95), 1),
                    "total_ms_p50": round(statistics.median(totals), 1),
                    "total_ms_p95": round(pct(totals, 0.95), 1),
                }
            )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()
    out = asyncio.run(main(args.runs))
    print(json.dumps(out, indent=2))
    print()
    print(
        "| upstream rate | config | TTFT p50 (ms) | TTFT p95 (ms) "
        "| total p50 (ms) | total p95 (ms) |"
    )
    print("|---|---|---|---|---|---|")
    for r in out["rows"]:
        print(
            f"| {r['rate_chars_per_s']} chars/s | {r['config']} | {r['ttft_ms_p50']} | "
            f"{r['ttft_ms_p95']} | {r['total_ms_p50']} | {r['total_ms_p95']} |"
        )
