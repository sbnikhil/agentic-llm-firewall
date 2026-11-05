# 0003: Plug into LiteLLM as a guardrail instead of building a gateway

- Status: accepted
- Date: 2026-11-19

## Context

The firewall has to sit between callers and the model on every request and every response, and it has to know who the caller is, because the policy depends on it. That needs:

- an OpenAI-compatible endpoint that applications already know how to call;
- per-user and per-team API keys, with revocation, budgets and rate limits;
- routing to many model providers;
- streaming;
- a place to run checks before the request goes upstream and on the response before the client sees it.

The roadmap's first engineering standard is that the firewall plugs into the gateway companies already run, and v1 explicitly excludes a custom gateway, key system or login system (docs/ROADMAP.md, sections 1 and 3). LiteLLM's proxy is that gateway. Its per-user keys are the identity.

## Options

1. **Build our own gateway** (FastAPI in front of the providers).
   - Pro: full control of the request path, no third-party internals to track.
   - Con: we would own key management, auth, provider adapters, streaming formats, budgets and the admin UI. These are security-critical, and none of them is what this project is about. Our key system would be one more thing a company has to trust and operate. v0 did a version of this and it stayed a demo.
2. **A separate proxy in front of, or behind, LiteLLM** (a sidecar that inspects HTTP).
   - Pro: independent of LiteLLM's Python internals.
   - Con: it cannot see LiteLLM's verified identity, only the raw key, so it would have to repeat LiteLLM's key lookup or trust a header. Behind LiteLLM, it would see provider-specific request formats. Streaming would need a second SSE parser and re-encoder.
3. **A LiteLLM `CustomGuardrail`** (chosen). The firewall runs inside the proxy process as a callback class.
   - Pro: the hooks receive `UserAPIKeyAuth`, the identity LiteLLM verified from the key's database record, before the request leaves. One hook sees the request, one sees the non-streaming response, and one wraps the response stream. Companies that run LiteLLM add one block to their config.
   - Con: we depend on LiteLLM internals that change often (a release every few days). Hook signatures, how hooks are dispatched and how the guardrail is gated must be pinned and tested. The firewall's dependencies (torch, spaCy, Presidio) must be installed next to LiteLLM's.
4. **Other gateways** (Envoy/Kong AI gateways, Portkey).
   - Con: not the gateway the roadmap targets. The same arguments as option 3 would apply, with less Python-native integration for our Python engine.

## Decision

Option 3: `llmfirewall.adapters.litellm_guardrail.LLMFirewallGuardrail`, a `CustomGuardrail` subclass, pinned to `litellm==1.108.3`. The risk of depending on LiteLLM internals is handled by pinning the version, by a contract test that fails CI when a hook changes, and by overriding the one piece of LiteLLM behaviour that would weaken the firewall (`should_run_guardrail`, below).

## What was verified in litellm 1.108.3

Read in the installed package (`.venv/lib/python3.12/site-packages/litellm/`) and pinned by `tests/unit/test_litellm_contract.py`, which compares parameter names, order, kinds and defaults with `inspect.signature`.

| Hook | Signature | Dispatch (proxy/utils.py) |
|---|---|---|
| `async_pre_call_hook` | `(self, user_api_key_dict, cache, data, call_type)` | `ProxyLogging.pre_call_hook` → `_process_guardrail_callback`. A returned dict replaces the request; an exception rejects it. |
| `async_post_call_success_hook` | `(self, data, user_api_key_dict, response)` | `post_call_success_hook`. A returned value replaces the response. **Not called for streams**: streaming routes return before it, and the deferred post-stream path (common_request_processing.py, around line 4061) skips any guardrail that defines the streaming iterator hook. |
| `async_post_call_streaming_iterator_hook` | `(self, user_api_key_dict, response, request_data)` | Wraps the whole chunk iterator. Detected only if defined on the **leaf class** (`cls.__dict__`, `callback_capabilities`), so the adapter defines it directly. |
| `apply_guardrail` | `(self, inputs, request_data, input_type, logging_obj=None)` | If a subclass defines it, the proxy routes pre_call, post_call and streaming through LiteLLM's "unified guardrail" instead of the hooks above (`_execute_guardrail_hook`, line 2364). **The adapter does not define it**, and a test checks that. |

**Identity.** `UserAPIKeyAuth.user_id` and `.team_id` come from the virtual key's database record, set by `user_api_key_auth` after the key is checked. Two cases:
- Calls made with the master key get `user_id="default_user_id"` (`litellm.constants.LITELLM_PROXY_ADMIN_NAME`) and no team, so they are unmapped and get the default policy.
- JWT auth, custom auth and OAuth2-proxy auth are the only other ways to set `user_id`, and none of them is enabled in `deploy/compose`. Enabling one changes what "verified" means and needs its own review.

The adapter reads only these two fields, through `VerifiedIdentity.from_litellm_auth`, as designed in docs/design/policy.md section 3.

**Gating, and why the adapter overrides it.** Before every hook, the proxy calls `callback.should_run_guardrail(data, event_type)`. LiteLLM's implementation lets things outside the proxy config switch a guardrail off:
- with `default_on: false`, the guardrail runs only if the request's `guardrails` list names it, and the client writes that list;
- with `default_on: true`, it is skipped if key or team metadata sets `disable_global_guardrails` or `opted_out_global_guardrails`, which an admin can set per key or per team.

The adapter overrides `should_run_guardrail` to return `True` always, so only the proxy config can remove the firewall. `tests/unit/test_litellm_guardrail.py` shows that the inherited method does return `False` for such metadata. The e2e suite sends the `guardrails` field, `disable_global_guardrails` (in the request body, in request metadata, in faked `user_api_key_*` metadata, and in real key and team metadata set through the admin API) and spoofed pipeline slots, and checks that each one is still blocked.

Other skip paths checked:
- `pipeline_managed_guardrail_names` reads `_guardrail_pipelines` from request metadata, but the proxy strips those slots from caller input before its policy engine writes them (`litellm_pre_call_utils.py`, `_strip_router_reserved_metadata`, line 2408).
- The "already ran" markers (`post_call_already_checked`, `mark_pre_call_hook_ran`) include a per-process random token (`secrets.token_hex(16)`), so a client cannot forge them.
- `_litellm_raw_sse_stream` only changes how the proxy serializes chunks after our iterator has yielded them.

**Call types and routes.** Two layers keep everything except chat completions away from the upstream (tests in `tests/e2e/test_routes.py`).

1. **Keys.** Every key is created with `allowed_routes` limited to the chat-completion routes: `/v1/chat/completions`, `/chat/completions`, `/engines/chat/chat/completions` and `/openai/deployments/chat/chat/completions`. bootstrap does this for alice and bob, and checks what LiteLLM stored.
   - LiteLLM enforces this in `RouteChecks.is_virtual_key_allowed_to_call_route` (`proxy/auth/route_checks.py`), before the request reaches any handler.
   - An explicit entry matches by exact path or by path prefix (`_route_matches_allowed_route`), or by a trailing `*`. A `{model}` placeholder is **not** expanded for explicit entries, so the aliases are listed with the model name.
   - An empty or missing `allowed_routes` means every route.
   - With restricted keys, LiteLLM returns 403 on `/v1/messages`, `/v1/responses`, `/v1/embeddings`, `/v1/completions`, `generateContent`, `/cursor/chat/completions`, `/milvus/*` and every provider pass-through prefix in `proxy/_lazy_features.py` (`llm_passthrough` and `anthropic_passthrough`, 32 prefixes; 40 routes tested in all).
2. **The guardrail.**
   - A key whose `allowed_routes` is unrestricted is blocked (`403 "key has no allowed_routes"`). This includes the master key.
   - Only `completion` and `acompletion` call types are checked; everything else is blocked with "unsupported call type".
   - Pass-through routes call the pre_call hook with call type `pass_through_endpoint` or `allm_passthrough_route`. This was verified by running them, not only by reading the code: with test-only fake Anthropic and OpenAI credentials pointed at the mock (`deploy/compose/docker-compose.e2e.yml`), `/anthropic/*` and `/openai_passthrough/*` pass LiteLLM's credential check and are then blocked by our hook. The mock receives nothing.
   - Without those credentials, LiteLLM rejects the routes before the hook, so the tests fail; this was checked.

**Chat aliases.**
- `/engines/{model}/chat/completions` and `/openai/deployments/{model}/chat/completions` are LiteLLM's chat-completion handler under other paths (call type `acompletion`). They are guarded chat routes and get every check: injection block, redaction, non-streaming leak block, streaming `content_filter` stop.
- `/cursor/chat/completions` (`proxy/response_api_endpoints/endpoints.py`) is handled differently, so it is **not** in the dev keys' `allowed_routes`:
  - a dependency that runs before auth can rewrite the model name;
  - a body with `messages` goes to the chat-completion handler after its `tools` field is rewritten, and gets every check;
  - a Responses-shaped body (`input`) goes through the Responses pipeline as `aresponses`, and our hook blocks it as an unsupported call type;
  - every other `/cursor/*` path is the Cursor Cloud Agents pass-through.

**Not verified:** MCP and realtime (WebSocket) routes. Restricted keys cannot call them.

**Import.** Importing litellm downloads its model cost map from GitHub unless `LITELLM_LOCAL_MODEL_COST_MAP=True`. The test suites set it (`tests/conftest.py`), and so does the compose stack.

## Postgres

LiteLLM's virtual keys, users and teams need a database. Both pages below were read on 2026-11-19. The docs site is not versioned (at that date it mentions releases up to v1.111), so the behaviour was also checked in the pinned code, where key auth without a database fails with "No connected db.".

- https://docs.litellm.ai/docs/proxy/virtual_keys, "Requirements": "Need a postgres database (e.g. Supabase, Neon, etc)" and "Set `DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<dbname>` in your env".
- https://docs.litellm.ai/docs/proxy/users, callout "Budgets require a database": "Key, team, and user budgets are unavailable for the same reason, since virtual keys cannot be resolved without a database (`No connected db.`)."

`deploy/compose` therefore runs Postgres 17 and the `litellm-database` image.

## Version pin and supply chain

**`litellm==1.108.3`, an exception to the two-week rule.** The plan was the newest stable release at least two weeks old, which was 1.108.1 (released 2026-11-05).
- With 1.108.1, pip-audit failed. 1.108.1 requires `filelock<4.0`, which moved filelock from 4.2.12 back to 3.32.7.
- 3.32.7 has 13 known vulnerabilities, all fixed only in 4.x: CVE-2026-112907, -112908, -112909, -112910, -112911, -112912, -112913, -112915, -112935, -112936 (fixed in 4.0.9), -112938, -112939 and -112940. They concern symlink, permission and stale-lock handling in filelock's soft and SQLite locks.
- 1.108.3 (released 2026-11-07, so 12 days old when pinned) relaxes the cap to `filelock<5`. That is the **only** core dependency difference from 1.108.1, compared in the PyPI metadata of both releases.
- With 1.108.3, filelock is back at 4.2.12 and `pip-audit --strict` over the whole lock finds no known vulnerabilities.

**Accepted downgrade.** `python-dateutil` went from 4.0.0 to 2.9.0.post0, because botocore (through boto3, a litellm dependency) requires 2.x. pip-audit does not flag it. With it, `tzdata` is no longer installed on macOS and Linux; it stays in the lock for pandas on Windows and emscripten.

**`openai==3.56.0`** is a dev dependency for the e2e suite, at the version the lock already resolved through litellm.

**Images**, pinned by tag and digest in `deploy/compose`:
- `ghcr.io/berriai/litellm-database:v1.108.3@sha256:aecfcc45…`
- `postgres:17.11-alpine@sha256:e5d2edd2…`
- `ollama/ollama:0.40.14@sha256:ebaef4f4…` (optional profile)

## How the firewall is installed in the LiteLLM image

`deploy/compose/proxy/Dockerfile` keeps LiteLLM's environment (`/app/.venv`, Python 3.13) exactly as shipped:
- It installs our package and its locked runtime dependencies from `uv.lock` into a separate venv, `/opt/llmfw`.
- A `.pth` file appends that venv to the **end** of LiteLLM's `sys.path`.
- For a package both have, LiteLLM's copy wins. `/opt/llmfw` supplies only what LiteLLM lacks: torch, spaCy, Presidio, sentence-transformers, transformers and `llmfirewall`.

Why: our `pyproject.toml` pins exact versions, including the frozen v0 app's dependencies (streamlit, langchain and others). Installing it into LiteLLM's environment would replace LiteLLM's own fastapi, pydantic and openai with versions LiteLLM was not released with.

Consequence: inside the proxy, the engine runs with some shared packages at LiteLLM's versions, not ours (checked in the running container):

| Package | uv.lock (tests) | Proxy image (runtime) |
|---|---|---|
| pydantic | 2.14.3 | 2.13.4 |
| pydantic-settings | 2.16.3 | 2.14.2 |
| numpy | 2.4.6 | 2.4.4 |
| fastapi | 0.165.0 | 0.136.3 |
| huggingface-hub | 1.33.0 | 1.32.0 |
| tokenizers | 0.23.3 | 0.23.1 |
| openai | 3.56.0 | 2.33.0 |

The unit and integration suites run on the lock. The e2e suite runs against the image, so it is what covers this combination. The CI pip-audit job audits `uv.lock` only, not the packages LiteLLM's image ships.

## Consequences

- Upgrading LiteLLM is a deliberate change: bump the pin and the image digests, run the contract test, re-check the dispatch facts above, and run e2e.
- The firewall cannot be turned off per request or per key or team, only by removing it from the proxy config.
- Tool-call arguments in responses pass unchecked, as already recorded (docs/design/policy.md section 10; v2).
- The compose config sets every relevant LiteLLM logging setting explicitly (`deploy/compose/litellm/config.yaml`, each commented with where litellm 1.108.3 reads it):
  - `LITELLM_LOG=WARNING`;
  - `turn_off_message_logging: true`;
  - `redact_messages_in_exceptions: true`. Without it, upstream exception text includes the request's messages (`exception_mapping_utils.py`);
  - `log_raw_request_response: false` and `set_verbose: false`;
  - `store_prompts_in_spend_logs: false`.

  After the e2e runs, the proxy log and all spend-log rows contained no request or response text.
- Every key must carry `allowed_routes`; the guardrail refuses keys without it, including the master key for model calls.
- LiteLLM's `/milvus/*` pass-through echoes the request body in its 400 error before any hook runs (docs/design/policy.md section 10, with a reproduction to report upstream).

Follow-up work:
- audit the proxy image's own packages, not only `uv.lock`;
- check MCP and realtime routes;
- pin the sentence model by revision. It is downloaded at image build without one, as in v0.

## Evidence

- `tests/unit/test_litellm_contract.py`: hook signatures, leaf-class definition, no `apply_guardrail`, `UserAPIKeyAuth` fields, the master-key sentinel, the pinned version.
- `tests/unit/test_litellm_guardrail.py`:
  - gating can't be switched off (every event type × 9 request/metadata variants);
  - every non-chat `CallTypes` value is blocked;
  - client-set fields can't choose the policy;
  - error and audit bodies contain no text.
- `tests/e2e/test_proxy.py` and `tests/e2e/test_routes.py` (155 tests, passing 3 out of 3 runs on 2026-11-19 after `docker compose -f docker-compose.yml -f docker-compose.e2e.yml up` from empty volumes):
  - two users get two policies;
  - client-set metadata, `guardrails` and `disable_global_guardrails` (request, key and team) do not bypass;
  - the dev keys are rejected on 40 non-chat routes, including every provider pass-through prefix and `/milvus/*`;
  - the chat aliases get every check;
  - unrestricted keys and the master key are blocked;
  - `/anthropic/*` and `/openai_passthrough/*`, with fake credentials, are blocked by the hook before reaching the mock;
  - a forced internal error returns a 500 without exception text.
- pip-audit, run as in CI (`uv export … | pip-audit --strict`): 13 filelock findings with 1.108.1; "No known vulnerabilities found" with 1.108.3.
