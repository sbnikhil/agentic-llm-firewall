# 0001: Freeze the v0 RAG app in v1; replace it in v2

- Status: accepted
- Date: 2026-11-19

## Context

v0 is one RAG application: a Streamlit UI, a FastAPI `POST /chat`, a LangGraph state machine, AstraDB retrieval and a local Ollama model. The firewall checks (input rules, Presidio PII redaction, output leak filter) run as nodes inside it.

The v1 goal (ROADMAP §1 and §3) is different: a guardrail that plugs into the LiteLLM gateway companies already run. LiteLLM's per-user keys are the identity. Answering from company documents is explicitly out of v1 and belongs to v2 (ROADMAP §4).

Facts that force the decision now:
- **The v0 app's main problems are architectural, not bugs to patch** ([v0-findings.md](../v0-findings.md)):
  - the caller picks their own role in the request body, and there is no authentication (finding 1);
  - raw queries and answers are written to the audit log (finding 3);
  - LangGraph state carries over between turns and between callers that share a `session_id` (finding 5);
  - LLM errors are reported as "Passed" (finding 2).

  Fixing these inside v0 would mean building a login and role system, which v1 rules out.
- **It cannot run in CI or on a clean machine without outside services** (AstraDB, Ollama), so it cannot be covered by the blocking CI that v1 requires.
- **Its detection code was tied to the app:** models loaded at import, and the output filter re-encoded the whole secrets catalog on every call (warm p50 about 213 ms; [baseline.md](../baseline.md)).
- **Its behavior is now pinned.** Characterization tests (unit, offline; and integration, real models, including every golden case) record exactly what the detectors do, so the detectors can be moved out and shown not to change.

## Options

1. **Keep v0 running and fix it in place.**
   - Pro: there is always a working demo.
   - Con: it needs auth, roles, log scrubbing and state fixes, all of which v1 replaces anyway. It also keeps AstraDB and Ollama in CI.
2. **Delete v0.**
   - Pro: the smallest repo.
   - Con: it loses the "what changed and why" story. The findings would then point at code that no longer exists, apart from git history.
3. **Move the detection code into the engine with the same behavior, and freeze the rest under `legacy/v0/`.**
   - Pro: the useful part (the detectors) continues, under tests. The rest stays readable and linkable.
   - Con: dead code in the tree, and v0-only dependencies stay in `pyproject.toml` until a later cleanup.

## Decision

Option 3. The detectors move unchanged into `src/llmfirewall/engine/detectors/`, behind `settings.py` and a load-once `models.py`. The app, UI, ingestion scripts, AstraDB code, training scripts and the compose file move to `legacy/v0/` unchanged, with a README saying the code is frozen. This keeps the evidence and drops the parts v1 replaces by design.

## What v2 replaces it with

ROADMAP §4.1 calls this "answers from documents, with real permissions". It is a retrieval example app built on top of the v1 engine and the LiteLLM guardrail, not a separate firewall:
- documents **and their permissions** come from a real source system (GitHub repositories first, Google Drive as an alternative), synced by a connector;
- a user only gets answers from documents they can open in that source system. This replaces v0's caller-chosen `user_role` and its hand-assigned `public/internal/confidential` levels;
- identity comes from what LiteLLM verified, never from the request body;
- it is measured on how long a revoked permission takes to stop affecting answers, and on the latency retrieval adds.

## Consequences

- Easier: CI tests only code that ships, with no outside services. The engine imports without loading models. Detector changes are now measured against characterization tests and the benchmark.
- Harder: there is no runnable end-to-end demo until the v1 LiteLLM guardrail and `deploy/compose/` exist. The Docker image currently only checks that the engine imports.
- Follow-up:
  - remove v0-only dependencies (FastAPI, LangChain/LangGraph, AstraDB, Streamlit, uvicorn, requests) from `pyproject.toml`;
  - replace the README's v0 architecture section once the v1 guardrail exists;
  - build the v2 retrieval app after v1 ships.
- `docs/v0-findings.md` links to the code at commit `9311a98` on GitHub, so its line references stay correct after the move.

## Evidence

- Findings: [docs/v0-findings.md](../v0-findings.md). The line links are permalinks to commit `9311a98`.
- Latency before and after the move: [docs/baseline.md](../baseline.md), "Latency (before)" and "After load-once". Both come from `.venv/bin/python bench/baseline_v0.py` on the same machine. Warm output-filter p50 went from 212.9 ms to 7.3 ms.
- Behavior unchanged: `uv run pytest tests/unit` (79 passed, 13 xfailed before and after, plus 2 new import-time tests) and `uv run pytest tests/integration -m integration` (57 passed before and after, including `test_characterization_golden.py`, which pins every golden redaction, pass and audit_fail case).
