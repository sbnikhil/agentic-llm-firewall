# v0 findings: bugs that will not be fixed in the v0 app

The v0 app (`app/`: FastAPI + LangGraph + AstraDB + Ollama) will be frozen as legacy code under `legacy/v0/` and will not be run. The bugs below are recorded, not fixed. Each entry gives the file and line, what happens, and which part of the v1 design (see [ROADMAP.md](ROADMAP.md), section 2) prevents it. Where the roadmap doesn't name a specific guard, the entry says so.

Line numbers refer to commit `9311a98`.

## 1. The caller picks their own role

- **Where:** [app/main.py:17](../app/main.py#L17) (`user_role: str = "employee"` in the request body), [app/main.py:82-83](../app/main.py#L82-L83), [app/main.py:93](../app/main.py#L93).
- **What happens:** `user_role` is a field in the JSON body of `POST /chat`. The only check is that it is one of `employee`, `manager` or `executive`. Any caller can send `"user_role": "executive"`. That value then decides the permission check in [app/graph/nodes.py:40](../app/graph/nodes.py#L40) and [app/graph/nodes.py:54](../app/graph/nodes.py#L54), and the retrieval filter in [app/graph/nodes.py:74-80](../app/graph/nodes.py#L74-L80). There is no authentication.
- **What prevents it in v1:** `engine/identity.py` maps the user or team that LiteLLM verified to a policy name, on the server side only. Nothing in the request body is read for identity. The v1 definition of done requires tests proving that a caller cannot choose or escalate their own policy.

## 2. An LLM error returns 200 "Passed"

- **Where:** [app/graph/nodes.py:130-137](../app/graph/nodes.py#L130-L137), [app/graph/workflow.py:58](../app/graph/workflow.py#L58), [app/graph/nodes.py:163-168](../app/graph/nodes.py#L163-L168), [app/main.py:130](../app/main.py#L130).
- **What happens:**
  1. When the Ollama call fails, `llm_node` returns a fallback text and sets `blocked: True` and `block_reason: "LLM service error: ..."`.
  2. The edge from `llm` to `output_security` is unconditional, so the output check still runs, on the fallback text.
  3. That text is harmless and passes, so `output_security_node` returns `blocked: False` and overwrites the error.
  4. `/chat` then returns HTTP 200 with the fallback as `ai_response` and a hard-coded `"audit_status": "Passed"`.

  The metrics record `blocked=False` together with the "LLM service error" reason ([app/main.py:110-111](../app/main.py#L110-L111)). The characterization test `plain-llm-error-fallback` in `tests/data/characterization/output_harmless.json` pins that the fallback text passes the output filter.
- **What prevents it in v1:** `engine/decision.py` combines findings and policy into allow, redact or block, and a decision "only ever gets stricter", so a later check cannot turn a block back into an allow. In v1 the model call itself belongs to LiteLLM, and the guardrail (`adapters/litellm_guardrail.py`) only inspects requests and responses. A model error is LiteLLM's error response, not a "Passed" from the firewall.

## 3. Raw query and response text in the audit log

- **Where:** [app/graph/nodes.py:175](../app/graph/nodes.py#L175) and [app/graph/nodes.py:178](../app/graph/nodes.py#L178), [app/security/audit_logger.py:28-31](../app/security/audit_logger.py#L28-L31), written to `security_audit.log` at [app/security/audit_logger.py:11-12](../app/security/audit_logger.py#L11-L12).
- **What happens:** every request writes `query` (the unredacted user input, including any PII or injected text) and `redacted_query` to a plain-text JSON-lines file. Whenever the answer was not blocked, it also writes the full model `response`. Blocked inputs are logged raw as well, because the input-block path sets `redacted_input` to the raw input ([app/graph/nodes.py:47](../app/graph/nodes.py#L47)).
- **What prevents it in v1:** `engine/audit.py` emits audit events containing types, offsets, counts and timings, and never raw text. The definition of done states: "No raw prompt or response text is written anywhere."

## 4. Exception text in 500 responses and `/health`

- **Where:** [app/main.py:135-138](../app/main.py#L135-L138) (500 detail), [app/main.py:46-51](../app/main.py#L46-L51) (AstraDB), [app/main.py:62-64](../app/main.py#L62-L64) (Ollama).
- **What happens:** any unexpected exception in `/chat` is returned to the caller as `"Internal processing error: <str(e)>"`. `/health` needs no authentication and returns `str(e)` for failed AstraDB and Ollama checks. Exception text can expose internal hostnames, endpoints, library versions and stack details.
- **What prevents it in v1:** v1 has no HTTP API of its own. The guardrail runs inside the LiteLLM proxy (`adapters/litellm_guardrail.py`), so LiteLLM owns the endpoints, error responses and health checks. The roadmap does not otherwise name a rule about error messages. That rule needs to be written for the adapter.

## 5. State carries over between turns (and between callers)

- **Where:** [app/graph/workflow.py:79-80](../app/graph/workflow.py#L79-L80) (`MemorySaver` checkpointer), [app/main.py:85-88](../app/main.py#L85-L88) (thread id = client-supplied `session_id`), [app/main.py:90-96](../app/main.py#L90-L96) (initial state sets only five keys).
- **What happens:** LangGraph merges each new input into the state saved for that `thread_id`. The initial state does not reset `block_reason`, `leak_score`, `passed_output_security`, `rag_context`, `llm_response`, `redacted_input` or `topic_level`, so values from an earlier turn remain. Example: turn 1 is blocked ("Prompt injection detected"), and turn 2 is a harmless question that passes. Turn 2's metrics record `blocked=False` with `block_reason="Prompt injection detected"` ([app/main.py:111](../app/main.py#L111)). Input-blocked turns never set `leak_score`, so they report the previous turn's score ([app/main.py:113](../app/main.py#L113)). Because `session_id` comes from the client, two callers who send the same `session_id` share one state. I confirmed the merge behavior with a toy LangGraph graph built the same way (MemorySaver, partial initial state). I did not run the app itself, which needs Ollama and AstraDB.
- **What prevents it in v1:** the engine is a library called once per request (`Engine.check_request(...)` / `Engine.check_response(...)`), with no conversation checkpointer. Each check sees only its own request or response. The roadmap describes this per-call API but does not state "stateless" as a rule, so a test should pin it when the engine exists.

## 6. The full metrics history is rewritten on every request

- **Where:** [app/metrics.py:36-41](../app/metrics.py#L36-L41).
- **What happens:** `record_request` appends the event to an in-memory list. `_save_metrics` then appends `json.dumps(self.metrics)`, the entire history so far, as a new line of `metrics.json`. After *n* requests the file holds 1 + 2 + … + *n* = *n*(*n*+1)/2 event copies, so it grows quadratically and each write gets slower. The in-memory list is never trimmed. Each event also includes `user_id`.
- **What prevents it in v1:** `engine/audit.py` emits one event per decision (types, counts, timings). The roadmap does not specify where metrics are stored, so the guard here is the design (one append per decision, never a snapshot) rather than a named component.

## Detection-code finding: the keyword-plus-similarity branch can never fire

This one is in detection code, not the app. Detection code moves into the engine during the restructure, but this branch is recorded as a finding, not as behavior to restore.

- **Where:** [app/security/output_filter.py:84](../app/security/output_filter.py#L84), [app/security/output_filter.py:103](../app/security/output_filter.py#L103), [app/security/output_filter.py:109-110](../app/security/output_filter.py#L109-L110).
- **What happens:** `validate_output` is meant to block when a sensitive keyword appears and similarity to a catalog secret is above 0.65 (comment at line 105). However, `check_semantic_similarity` returns a score of `0.0` whenever the best match is at or below `LEAK_THRESHOLD` (0.75). So in the only case where line 109 matters (not a semantic leak), `similarity` is always 0.0, and `similarity > 0.65` is never true. Example with the real MiniLM: "The CEO salary is set by the board each year." contains the keyword "salary" and scores 0.722 against "CEO salary is $2.5M annually", but `validate_output` returns `(True, "Passed", 0.0)`.
- **How it is pinned:** `tests/unit/test_characterization_output.py` (fake model at similarity 0.70) and `tests/integration/test_characterization_real_models.py` (real MiniLM, three sentences between 0.65 and 0.75). Both assert today's behavior, with no xfail. Any change must be a deliberate decision.
- **What prevents it in v1:** in v1, detectors report findings with their scores to `engine/decision.py`, which applies the policy. A combined rule like this one would be written there once, against scores the detectors actually report.
