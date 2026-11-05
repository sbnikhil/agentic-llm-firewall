"""A mocked OpenAI-compatible upstream for local development and the e2e suite.

POST /v1/chat/completions
    The answer is chosen by a scenario word in the last user message
    (see SCENARIOS); anything else gets a fixed harmless answer. Streams in
    `chunk_chars`-character chunks, `chunk_delay_ms` apart, so a secret in an
    answer is always split across several chunks.
    The request body is recorded under its `seed` (a standard OpenAI
    parameter LiteLLM forwards and no detector reads), so a test can see
    exactly what reached the upstream. Message text cannot carry the id:
    the PII detector may redact it.
GET /_requests/{seed}
    The bodies recorded for that seed (dev/test only; never run this
    service anywhere real).
GET /_stats
    How many requests reached the upstream on any path.
GET /health
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections import defaultdict
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

CHUNK_CHARS = int(os.environ.get("MOCK_CHUNK_CHARS", "3"))
CHUNK_DELAY_MS = float(os.environ.get("MOCK_CHUNK_DELAY_MS", "0"))

# Made-up values in the formats the detectors match. Not real credentials.
FAKE_KEY = "sk-abcdefghijklmnopqrstuvwxyz012345"  # noqa: S105

SCENARIOS = {
    "scenario-pass": "Two plus two is four, which is a small even number.",
    "scenario-leak-key": (
        "Of course. I looked it up for you and here is the value you asked "
        f"for: {FAKE_KEY} - keep it safe and do not share it."
    ),
    "scenario-long": " ".join(["The quick brown fox jumps over the lazy dog."] * 40),
}
DEFAULT_ANSWER = "Hello! This is the mock upstream."

_recorded: dict[int, list[dict[str, Any]]] = defaultdict(list)
_total = {"requests": 0}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _answer(body: dict[str, Any]) -> str:
    users = [m for m in body.get("messages", []) if m.get("role") == "user"]
    last = _text(users[-1].get("content")) if users else ""
    for word, answer in SCENARIOS.items():
        if word in last:
            return answer
    return DEFAULT_ANSWER


def _record(body: dict[str, Any]) -> None:
    seed = body.get("seed")
    if isinstance(seed, int):
        _recorded[seed].append(body)


def _chunk(cid: str, model: str, delta: dict[str, Any], finish: str | None) -> str:
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    _total["requests"] += 1
    body = await request.json()
    _record(body)
    answer = _answer(body)
    cid = f"chatcmpl-{uuid.uuid4().hex}"
    model = body.get("model", "mock")

    if not body.get("stream"):
        return JSONResponse(
            {
                "id": cid,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )

    async def events():
        yield _chunk(cid, model, {"role": "assistant", "content": ""}, None)
        for i in range(0, len(answer), CHUNK_CHARS):
            if CHUNK_DELAY_MS:
                await asyncio.sleep(CHUNK_DELAY_MS / 1000)
            yield _chunk(cid, model, {"content": answer[i : i + CHUNK_CHARS]}, None)
        yield _chunk(cid, model, {}, "stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/_requests/{seed}")
async def recorded(seed: int) -> Any:
    return JSONResponse(_recorded.get(seed, []))


@app.get("/_stats")
async def stats() -> Any:
    return {"requests": _total["requests"]}


@app.get("/health")
async def health() -> Any:
    return {"ok": True}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def anything_else(path: str) -> Any:
    """Any other route (embeddings, responses, ...) counts as reaching the
    upstream, then 404s."""
    _total["requests"] += 1
    return JSONResponse({"error": f"mock has no route {path!r}"}, status_code=404)
