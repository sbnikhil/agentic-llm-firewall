# LLM Firewall: roadmap (v3)

Project context for anyone working in this repo, including coding agents. It describes the goal, architecture and standards. It contains no task instructions: work is assigned one task at a time.

## 1. Goal

A security checkpoint between people and an AI model. It checks every message on the way in (prompt injection, secrets, personal info) and every answer on the way out (leaks), and logs every decision without storing the text.

Engineering standards:
1. it plugs into the gateway companies already run (LiteLLM)
2. it's fast, with a latency budget measured and enforced in CI
3. every reported number comes from independent public datasets, never from data written for this project
4. it is used by real people, and their usage drives fixes

**Not in v1:**
- answering questions from company documents (v2, section 4)
- a custom gateway, API-key system or login system (LiteLLM's per-user keys are the identity)
- PyPI or GHCR releases

## 2. Architecture

```
src/llmfirewall/
  engine/                  # pure library, nothing heavy at import
    settings.py            # pydantic-settings, env prefix LLMFW_
    policy.py              # policy schema, loaded from YAML
    identity.py            # verified LiteLLM user/team -> policy name, server-side only
    models.py              # models loaded once, cached, warmed at startup
    detectors/             # secrets, pii, injection, confidential_terms, ...
    pipeline.py            # cheap checks first; expensive checks only when needed
    decision.py            # Findings + Policy -> allow / redact / block; only ever gets stricter
    engine.py              # Engine.check_request(...) / Engine.check_response(...)
    audit.py               # audit events: types, offsets, counts, timings, never raw text
  adapters/
    litellm_guardrail.py   # LiteLLM CustomGuardrail: pre_call, streaming iterator, post_call
bench/                     # latency benchmarks and the CI latency gate
eval/                      # pinned independent datasets, contamination checks, harness, reports
train/                     # fine-tuning candidate detectors on dev-train data only
deploy/compose/            # LiteLLM proxy + Postgres + guardrail + chat UI
legacy/v0/                 # the original LangGraph/AstraDB app, frozen, not run, kept for the "what changed" story
tests/unit  tests/integration  tests/e2e
docs/  docs/decisions/
```

## 3. Definition of done (v1)

- `docker compose up` in `deploy/compose/` brings up LiteLLM with the guardrail and a chat UI on a clean machine, and the README quickstart works exactly as written.
- The latency budget is written down, measured, and enforced in CI.
- Every number in the README comes from `eval/`, on independent, pinned, de-duplicated datasets, with counts and confidence intervals.
- The policy is chosen only from the identity LiteLLM verified. Tests prove a caller cannot choose or escalate their own policy.
- No raw prompt or response text is written anywhere.
- CI blocks merges on lint, types, tests, the eval smoke set and the latency gate.
- `docs/usage-report.md` covers at least two weeks of real use by me and at least two other people.

## 4. v2 (after v1 ships)

1. **Answers from documents, with real permissions:** a retrieval example app whose documents and permissions come from a real source system (GitHub repos first; Google Drive as an alternative), synced by a connector, so a user only gets answers from documents they can open there. Measured: how long a revoked permission takes to stop affecting answers, and the added latency of retrieval.
2. **Tool-call checks for AI agents:** run the engine on `tool_calls` arguments (for example, block a `send_email` body containing confidential terms), evaluated on AgentDojo with benign utility, utility under attack and attack success rate.
3. **Business-confidential leak benchmark (research track):** role-aware ground truth, paraphrased facts, scored on leak rate and answer quality.
