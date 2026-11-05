# Design: policy, identity, findings, decisions, pipeline

- Status: **accepted**, 2026-11-19
- Modules: `src/llmfirewall/engine/{policy,identity,decision,pipeline,engine}.py`, `src/llmfirewall/engine/detectors/builtin.py`
- Tests: `tests/unit/test_{policy,identity,decision,pipeline,engine,builtin_detectors}.py`
- Related: [ADR 0002: fail closed](../decisions/0002-fail-closed.md)

This document describes how the engine turns detector output into one decision per request or response, how a policy is chosen for a caller, and what happens when a check fails or runs late. The existing detectors (`input_rules`, `pii`, `output_filter`) are wrapped, not changed.

Section 9 records the decisions made at review. Section 10 lists the known limitations.

---

## 1. Data flow

```
LiteLLM verified key ──► identity.py ──► Policy (one, or the stricter merge of two)
                                            │
request messages ──► engine.check_request(messages, policy)
                        │
                        ├─ split messages into segments (role, message_index, text)
                        ├─ pipeline: detectors cheapest first, clock checked after each call
                        │     each detector → Hits → Findings (no text)
                        ├─ decision: block > redact > allow, only ever stricter
                        └─ Decision(action, findings, failures, skipped, redacted copy)
```

`check_response(text, policy)` follows the same path with one segment whose role is `response`.

## 2. Policy YAML schema (`policy.py`)

A policy is one YAML file in `policies/`. The policy's name is the file name without `.yaml`. The schema is a set of pydantic models with `extra="forbid"`, `frozen=True` and `strict=True`, so unknown keys, wrong types and coerced values (for example the string `"true"` for a boolean) are all errors.

```yaml
version: 1                       # required; only 1 is accepted
on_detector_error: fail_closed   # fail_closed | fail_open; default fail_closed
on_timeout: fail_closed          # fail_closed | fail_open; default fail_closed
latency_budget_ms: 1000          # required; > 0. Budget for all checks on one call
detectors:                       # required; every registered detector must be listed
  input_rules:
    enabled: true                # required
    action: block                # allow | redact | block; required
    apply_to: [user]             # required; non-empty when enabled
    params: {}                   # optional; validated by the detector's params model
  pii:
    enabled: true
    action: redact
    apply_to: [user, response]
    params:
      entities: [PERSON, EMAIL_ADDRESS]   # subset of pii.REDACT_ENTITIES; default all
      min_score: 0.0                      # results below this score are dropped
  output_leak:
    enabled: true
    action: block
    apply_to: [response]
```

Field rules:

| Field | Type | Rule |
|---|---|---|
| `version` | `Literal[1]` | Missing or any other value is an error. |
| `on_detector_error` | `fail_closed \| fail_open` | Default `fail_closed`. What happens when a detector raises or returns invalid output (section 6). |
| `on_timeout` | `fail_closed \| fail_open` | Default `fail_closed`. What happens when the budget is exceeded (section 6). |
| `latency_budget_ms` | int, > 0 | Required, so nobody runs without a budget by accident. |
| `detectors` | map name → config | Keys must be exactly the registered detectors: an unknown name is an error, and so is a missing one. A newly added detector therefore forces every policy to decide whether it is on, instead of silently defaulting either way. |
| `enabled` | bool | Disabled detectors are never called. |
| `action` | `allow \| redact \| block` | `allow` records findings without acting on them (monitor mode). |
| `apply_to` | list of roles | Roles: `system`, `developer`, `user`, `assistant`, `tool`, `function` (request messages) and `response` (the model's answer in `check_response`). Unknown roles are errors. A request message whose role is `response` is unsupported input (section 6). |
| `params` | map | Validated by the detector's own params model, a `DetectorParams` subclass. Unknown keys are errors. |

Loading:
- Files are parsed with `yaml.safe_load`. Before that, the same text is composed with `yaml.compose(..., Loader=yaml.SafeLoader)`, which builds only the node tree and constructs no objects, and any **duplicate key** is rejected. Plain `safe_load` keeps the last duplicate silently, so `action: block` followed later by `action: allow` would quietly allow.
- Any problem raises `PolicyError` (a `ValueError`), with the file name and the field path. Policies are loaded at startup, so a bad policy stops the service from starting rather than being skipped.
- API: `parse_policy(data, *, name, detectors) -> Policy`, `load_policy_file(path, *, detectors) -> Policy`, `load_policies(directory, *, detectors) -> dict[str, Policy]`. `load_policies` reads every `*.yaml` in the directory except `identity.yaml`, and fails if any one of them is invalid.
- Settings: `LLMFW_POLICIES_DIR` (default `policies/`) and `LLMFW_IDENTITY_MAP_PATH` (default `policies/identity.yaml`). The LiteLLM adapter reads them at startup ([ADR 0003](../decisions/0003-litellm-guardrail.md)).

### Parameters of the existing detectors

The detectors are wrapped unchanged, and their thresholds (`max_query_length`, `max_special_char_ratio`, `leak_threshold`, `keyword_similarity_threshold`) are read from `settings.py` inside the detector code. A policy therefore cannot change them without changing the detectors. Only parameters the wrapper can apply to the detector's output are offered:

| Detector | Params | Applied by the wrapper |
|---|---|---|
| `input_rules` | none | — |
| `pii` | `entities` (subset of `REDACT_ENTITIES`, default all), `min_score` (0–1, default 0.0) | Filters Presidio's results after the same `analyze` call `redact_pii` makes. With the defaults, every result is kept. |
| `output_leak` | none | — |

Moving the thresholds into policy params is a later, separate change.

### Merging two policies

`merge_stricter(a, b)` builds one policy whose **effective behavior** is at least as strict as both. A disabled detector does nothing, so its `action`, `apply_to` and `params` have no effect and do not count in the merge. Per detector:

| Enabled in | Merged detector |
|---|---|
| both | enabled; the stricter `action` (block > redact > allow); the union of both `apply_to` lists; `params` merged by the params model's `stricter()`. For `pii` that is the union of `entities` and the lower `min_score`, since both find more. Params models without fields (`input_rules`, `output_leak`) are always equal. |
| one | that side's config, unchanged |
| neither | disabled, with neutral values that have no effect: `allow`, no roles, default params |

Policy-wide fields:

| Field | Merged value |
|---|---|
| `on_detector_error`, `on_timeout` | `fail_closed` if either is `fail_closed` |
| `latency_budget_ms` | the lower |

Example: a detector disabled with `action: block` in one policy and enabled with `action: allow` in the other merges to enabled with `allow`, on the enabled side's roles and with its params. The disabled side's `block` is not in effect anywhere, so carrying it over would make the merge stricter than either policy actually behaves.

The merged policy is named `<a>+<b>`, and that name appears in `Decision.policy_name`.

## 3. Identity binding (`identity.py`)

The policy is chosen only from identity that LiteLLM verified from the caller's virtual key. It is never taken from request content, request metadata or headers.

Mapping file (`policies/identity.yaml`, server-side, loaded at startup):

```yaml
version: 1
default: default            # required; used for every caller with no mapping
users:                      # optional; LiteLLM user_id -> policy name
  alice: strict
teams:                      # optional; LiteLLM team_id -> policy name
  contractors: strict
```

Resolution:
1. If both `user_id` and `team_id` are mapped, the caller gets `merge_stricter(user policy, team policy)`, so neither mapping can weaken the other.
2. If only one is mapped, the caller gets that policy.
3. Otherwise, the caller gets `default`.

Every user/team policy pair is merged when the resolver is built, so a merge problem stops startup rather than failing a request.

Validation at load: `default` is required; every policy name in the file must exist in the loaded policies; ids must be non-empty strings; unknown keys are errors; duplicate keys are rejected. Empty, missing or non-string ids from LiteLLM count as "no id".

API:

```python
class VerifiedIdentity(BaseModel):  # frozen, strict
    user_id: str | None
    team_id: str | None

    @classmethod
    def from_litellm_auth(
        cls, auth: Any
    ) -> VerifiedIdentity: ...  # reads auth.user_id, auth.team_id only


class PolicyResolver:
    def __init__(
        self, identity_map: IdentityMap, policies: Mapping[str, Policy]
    ) -> None: ...
    def resolve(
        self, identity: VerifiedIdentity
    ) -> Policy: ...  # TypeError for anything else
```

`resolve` takes only a `VerifiedIdentity`. There is no parameter through which request data, metadata or headers could reach it. The tests check that, and that `from_litellm_auth` reads no attribute other than `user_id` and `team_id`.

### Which LiteLLM fields count as verified

In LiteLLM, a `CustomGuardrail` hook receives `user_api_key_dict: UserAPIKeyAuth`, the result of the proxy checking the caller's key, alongside `data`, the request body.

| Source | Verified? | Why |
|---|---|---|
| `UserAPIKeyAuth.user_id` | **yes** | Set from the virtual key's database record (or from JWT claims when JWT auth is turned on), after the key is checked. |
| `UserAPIKeyAuth.team_id` | **yes** | Same. |
| `UserAPIKeyAuth.end_user_id` | no | Copied from the request body's `user` field, which the client sets. |
| `data["user"]`, `data["metadata"]`, `data["litellm_metadata"]` | no | Request body. |
| every HTTP header, including `x-litellm-*` | no | Client-set. |
| `UserAPIKeyAuth.metadata`, `team_metadata`, `key_alias`, `team_alias` | not used | Admin-set, but not identity. Kept out to keep the trust surface to two fields. |

Confirmed against `litellm==1.108.3` ([ADR 0003](../decisions/0003-litellm-guardrail.md)):
- `user_id` and `team_id` are fields of `UserAPIKeyAuth`, set from the key's database record before the `pre_call` hook runs;
- master-key calls get `user_id="default_user_id"` and no team, so they would be unmapped and get `default`. In practice the adapter blocks them, because the master key has no `allowed_routes` (below);
- the other ways to set `user_id` are JWT auth, custom auth and OAuth2-proxy auth, none of which is enabled in `deploy/compose`. Enabling one needs its own review.

**Route restriction.** The adapter also reads `UserAPIKeyAuth.allowed_routes`, only to fail closed and never to choose a policy. A key whose `allowed_routes` is unrestricted (None, an empty list or not a list, which LiteLLM treats as every route) is blocked with `403 "key has no allowed_routes"` before any other check. Every key must therefore be created with `allowed_routes` (deploy/compose/README.md, "Keys"). `VerifiedIdentity` still reads only `user_id` and `team_id`.

## 4. Findings (`decision.py`)

```python
class Span(BaseModel):  # frozen; 0 <= start < end; offsets in Python str indices
    start: int
    end: int


class Finding(BaseModel):  # frozen, extra="forbid"
    detector: str  # e.g. "pii"
    entity_type: str  # e.g. "EMAIL_ADDRESS", "PROMPT_INJECTION"
    score: float  # 0.0-1.0
    message_index: int | None  # index in the request messages; None for a response
    span: Span | None  # None: the detector flags the whole message, no location
    duration_ms: float  # time the detector call took for this message
```

A `Finding` has no field that can hold text, and `extra="forbid"` stops one from being added by accident. Its repr and JSON therefore cannot contain the matched text. A test pins the exact field set, so adding a field means changing that test on purpose.

Failures are recorded the same way, also without text:

```python
class DetectorFailure(BaseModel):  # frozen, extra="forbid"
    detector: str  # "engine" for unsupported_input
    kind: Literal["error", "timeout", "invalid_output", "unsupported_input"]
    error_type: str | None  # exception class name only, e.g. "ValueError"
    message_index: int | None
```

The exception message is **not** stored, because exception messages often include the input (`KeyError: 'alice@example.com'`).

Detectors return `Hit(entity_type, score, span)`. The pipeline turns each hit into a `Finding` by adding the detector name, the message index and the duration. The wrappers in `detectors/builtin.py` map today's outputs as follows:

| Detector | Cost | Hit | Span |
|---|---|---|---|
| `input_rules` | cheap | One hit when `validate_input` rejects. The reason maps to a fixed type: `EMPTY_INPUT`, `INPUT_TOO_LONG`, `CSV_INJECTION`, `PROMPT_INJECTION`, `SUSPICIOUS_PATTERN`. Score 1.0. | none |
| `pii` | expensive | One hit per Presidio result, from the same `models.presidio_analyzer().analyze(text=..., language="en", entities=REDACT_ENTITIES)` call `redact_pii` makes. | yes |
| `output_leak` | expensive | One hit when `validate_output` blocks. Type `SECRET_PATTERN`, `SECRET_SIMILARITY` or `SENSITIVE_KEYWORD` from the reason prefix. The score is the returned score, clamped to 0–1. `SENSITIVE_KEYWORD` cannot occur today, because that branch never fires (pinned in the characterization tests). | none |

Reason strings are mapped, never stored. An unrecognised reason raises `DetectorOutputError`, which the pipeline reports as `invalid_output`.

## 5. Decision rules (`decision.py`)

- `Action` is an ordered enum: `ALLOW < REDACT < BLOCK`. Combining actions takes the maximum, so block beats redact beats allow.
- Each finding gets its detector's `action` from the policy.
- **Only stricter:** the pipeline holds the running action in a `Verdict` whose only operation is `escalate(action)`, which keeps the maximum. Nothing can lower it. `Decision` is frozen.
- **Redaction without a location becomes a block.** If a detector whose action is `redact` reports a finding with no span, there is nothing to replace, so that finding counts as `block`. Today this applies if `input_rules` or `output_leak` is set to `redact`.
- **Placeholders** are `[REDACTED:<ENTITY_TYPE>]`, for example `[REDACTED:EMAIL_ADDRESS]`.
- **Overlapping spans:** spans that overlap (share at least one character) merge, transitively, into one span covering all of them, replaced by one placeholder. The label is the entity type of the highest-scoring finding in the group. Ties go to the alphabetically first type, so the output is deterministic. Spans that only touch (`a.end == b.start`) are not merged. Replacement happens right to left, so earlier offsets stay valid.
- A span past the end of the text is a detector bug. The pipeline checks spans against the text and reports `invalid_output`, which is handled like an error (section 6).
- Only findings from detectors whose action is `redact` are redacted. With `block` nothing is redacted, because nothing is forwarded.

```python
class Decision(BaseModel):  # frozen
    action: Action
    policy_name: str
    findings: tuple[Finding, ...]
    failures: tuple[DetectorFailure, ...]
    skipped: tuple[str, ...]  # enabled detectors not run (short-circuit or budget)
    redacted_messages: tuple[dict, ...] | None  # repr=False; only when REDACT
    redacted_text: str | None  # repr=False; only when REDACT
```

The redacted copy is the content to forward, so it lives on the `Decision`, but it is excluded from `repr` so that logging a decision does not log text. The audit module (`engine/audit.py`) builds its events from `findings`, `failures`, `skipped` and `durations_ms` only (per-detector time, added for the audit log; no decision reads it).

## 6. Pipeline (`pipeline.py`)

```python
class CostClass(IntEnum):
    CHEAP = 0  # pure Python, microseconds (input_rules)
    EXPENSIVE = 1  # model inference (pii, output_leak)


class Detector(Protocol):
    name: str
    cost: CostClass
    params_model: type[DetectorParams]

    def detect(self, text: str, params: DetectorParams) -> Sequence[Hit]: ...
```

**Order:** by `cost`, then by name, so the order is deterministic and does not depend on the YAML. Within a detector, segments run in message order. A detector runs only on segments whose role is in its `apply_to`.

**Short-circuit:** when the running action reaches `BLOCK`, the remaining detectors of the current cost class still run and all their findings are recorded. Every later cost class is skipped and listed in `Decision.skipped`. A `REDACT` never short-circuits, because later detectors may find more to redact or a reason to block.

**Errors:** a detector that raises, or returns something other than `Hit`s with spans inside the text, produces a `DetectorFailure` (`error` or `invalid_output`). Its hits for that segment are discarded.
- `fail_closed`: the running action escalates to `BLOCK`, and the short-circuit applies.
- `fail_open`: that detector contributes nothing for that segment; the failure is still recorded. A failure never lowers the action.

**Latency budget:** checked on the clock (`time.monotonic`, injectable for tests) after each detector call. There are no threads.
- The budget covers the whole call: `deadline = start + latency_budget_ms`. When a call finishes after the deadline, a `timeout` failure is recorded for that detector and message. No further call starts, and the detectors not yet started are listed in `skipped`. Finishing exactly at the deadline is not a timeout.
- The findings of the call that overran are kept. It finished, so its result is known, and dropping it could only make the decision less strict.
- `fail_closed`: `BLOCK`.
- `fail_open`: the decision from the findings so far. With `fail_open`, a PII timeout means unredacted text is forwarded.
- A single slow call is not interrupted. See known limitations (section 10).

**Request content the engine cannot check:**
- Message `content` may be a string, `None` or absent (for example an assistant message carrying only `tool_calls`; nothing to check, and tool-call arguments are v2), or a list of text parts. Text parts are joined with `"\n"` into one segment, so offsets refer to the joined string, and a redacted message gets string content.
- A non-text part (image, audio, file), any other content type, a missing or unknown role, or the role `response` in a request cannot be checked. The decision is `BLOCK` with a failure of `kind="unsupported_input"`, even under `fail_open`, and no detector runs. Letting it through would be a bypass: put the payload in an image.

## 7. Engine API (`engine.py`)

```python
class Engine:
    def __init__(
        self,
        detectors: Sequence[Detector] | None = None,
        *,
        clock: Clock = time.monotonic,
    ) -> None: ...

    # None -> builtin_detectors(): input_rules, pii, output_leak. Nothing loads here.

    def check_request(
        self, messages: Sequence[Mapping[str, Any]], policy: Policy
    ) -> Decision: ...
    def check_response(self, text: str, policy: Policy) -> Decision: ...
```

- `policy` must have been parsed against the same detectors as the engine (same names, same params models). A mismatch raises `PolicyError`, which is a configuration bug; the adapter will treat any exception as a block.
- The engine never reads a policy name from `messages`. It receives a `Policy` object resolved through `identity.py`.
- The input messages are not modified. `redacted_messages` holds copies, with only `content` replaced, and only in the messages that had something to redact.
- Importing any engine module loads no model, and neither does `Engine()`. The existing import-time test covers the new modules automatically.

## 8. Shipped policies

### `policies/default.yaml`

Matches v0's behavior, so the evaluation baseline measures v0 truthfully. Changes come later, from measurement.

| Choice | Value | Why |
|---|---|---|
| `on_detector_error` / `on_timeout` | `fail_closed` | [ADR 0002](../decisions/0002-fail-closed.md). |
| `latency_budget_ms` | `1000` | **Unmeasured placeholder.** For reference, warm p95s are 0.004 ms, 4.5 ms and 18 ms ([baseline.md](../baseline.md)), but PII runs once per message, so long chats cost more. The real number comes from the latency task. |
| `input_rules` | block, `[user]` | System and developer messages are written by the application holding the key, and the phrase rules would block ordinary system prompts ("act as if", "role play"). Tool output would trip the special-character ratio on JSON. See section 10. |
| `pii` | redact, `[user, response]`, all entities, `min_score: 0.0` | As in v0: redact, don't block. |
| `output_leak` | block, `[response]` | As in v0. It also blocks any answer containing an email or phone number before PII redaction gets a chance (pinned by the characterization tests). |

### `policies/strict.yaml`

| Choice | Value | Why |
|---|---|---|
| failure modes | `fail_closed` | As in default. |
| `latency_budget_ms` | `1000` | Unmeasured placeholder, as in default. |
| `input_rules` | block, `[user, assistant, tool, function]` | Every role a client writes. `assistant` because a client can fabricate earlier assistant turns; `tool` and `function` because tool output carries indirect injection. Not `system` or `developer`: the application writes those, and the phrase rules would block ordinary system prompts. |
| `pii` | redact, all request roles + `response` | Redaction already stops personal data passing in either direction. Block with `min_score: 0.0` would reject a request over any low-confidence Presidio match, such as a date. |
| `output_leak` | block, `[response]` | As in default. |

`policies/identity.yaml` ships with `default: default` and no mappings.

## 9. Decisions made at review (2026-11-19)

1. PyYAML 6.0.3 added as a direct dependency; files are parsed with `safe_load` only.
2. When both the user and the team are mapped, the stricter merge of the two policies (section 2), not user-wins. Tests prove a user mapping can never weaken the team policy's effective behavior. Corrected after the first implementation: a disabled detector's action, roles and params do not count in the merge.
3. Redact with no span becomes block.
4. Overlapping spans merge, labelled by the highest score, ties alphabetical.
5. After a block, the remaining detectors of the same cost class still run, and all findings are recorded.
6. Latency budget: a clock check between detector calls only, fail closed when exceeded. No threads now; a hard deadline comes with the latency task.
7. Unsupported content is always blocked, even under `fail_open`.
8. `default.yaml` checks `input_rules` on `[user]` only (section 10).
9. `latency_budget_ms: 1000` is an unmeasured placeholder.
10. `default.yaml` as proposed. In `strict.yaml`, `input_rules` covers `[user, assistant, tool, function]` and `pii` redacts instead of blocking.

## 10. Known limitations

- **`default.yaml` does not check system, developer, assistant or tool messages with `input_rules`.** Every role in a request is technically client-supplied, so an injection placed in a system message is not caught by the default policy. This is acceptable because the system prompt is written by the application holding the LiteLLM key, not by its end users. A caller who holds a key and writes their own system prompt is already trusted to configure the model. Applications that forward untrusted text into other roles should use `strict`.
- **`strict.yaml` does not check system or developer messages with `input_rules`**, for the same reason: the application writes them, and the phrase rules would block ordinary system prompts. `pii` still covers them in `strict`.
- **The latency budget is not a hard deadline.** The clock is checked after each detector call, so one slow call (for example a model stuck on a long input) runs to completion before the request fails closed. The latency task will add a hard deadline.
- **Detector thresholds are not policy parameters** (section 2). They stay in settings until the detectors are changed.
- **`output_leak` blocks any email or phone number in an answer**, so `pii` redaction of answers rarely matters under the shipped policies. This is v0 behavior, pinned on purpose.
- **Tool-call arguments are not checked** (v2). Specifically:
  - In requests, `tool_calls` on assistant messages are never read. An assistant message with only `tool_calls` (`content` is `None`) passes with nothing checked, and when it has both, only `content` is checked.
  - In non-streaming responses, `message.tool_calls` and `function_call` pass unchecked; only `message.content` is checked.
  - In streams, tool-call deltas are forwarded as they arrive. They are not held back by the window and not scanned.

  A model can therefore return a secret, or forward an injection, inside tool-call arguments.
- **The PII detector tags random hex and letter tokens as PERSON or LOCATION.** spaCy's named-entity recognition (Presidio's default NLP engine, `en_core_web_lg`) labels many random tokens as names or places.

  Measured on 2026-11-19 with the shipped analyzer (`models.presidio_analyzer().analyze(..., entities=REDACT_ENTITIES)`):
  - over 300 strings `"hello nonce-<uuid4 hex>"`, it returned 64 PERSON and 6 LOCATION results;
  - over 300 strings `"hello nonce<16 random lowercase letters>"`, it returned 183 PERSON and 22 LOCATION results.

  Under any policy where `pii` redacts (`user` and `response` in `default`, every role in `strict`), request ids, hashes, tokens and similar text are often replaced with `[REDACTED:PERSON]`. Clients and tests must not rely on such tokens surviving the firewall; the e2e suite correlates requests by the OpenAI `seed` parameter for this reason. This is v0 behavior (pinned by the characterization tests) and belongs to the evaluation work.
- **LiteLLM's `/milvus/*` pass-through echoes the request body in its error response.** Found in litellm 1.108.3, route `/milvus/{endpoint:path}` (`litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`, `milvus_proxy_route`, line 814: `detail=f"Collection name is required. Got {request_body}"`).

  A body without `collectionName` is returned verbatim in a 400. This happens before the pre_call hook, so the guardrail never sees the request. The text goes back to the caller; in our tests it was not written to the proxy log.

  Minimal reproduction, with a key allowed on `/milvus/*` and made-up data:

  ```bash
  curl -s -X POST http://localhost:4000/milvus/vectors/search \
    -H "Authorization: Bearer sk-<virtual key>" \
    -H 'Content-Type: application/json' \
    -d '{"note": "example-text"}'
  # 400 {"detail":"Collection name is required. Got {'note': 'example-text'}"}
  ```

  In `deploy/compose`, keys restricted to the chat routes cannot reach `/milvus/*`, and the e2e suite checks this. An unrestricted key would reach it, even though the guardrail blocks unrestricted keys, because this error comes before the hook.
