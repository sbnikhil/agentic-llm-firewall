# Compose stack: LiteLLM proxy + firewall guardrail

What runs:

| Service | What it is |
|---|---|
| `litellm` | LiteLLM's official proxy image (`litellm-database:v1.108.3`) with the firewall guardrail installed. Port `127.0.0.1:4000`. |
| `postgres` | Postgres 17. LiteLLM needs it for virtual keys, users and teams ([ADR 0003](../../docs/decisions/0003-litellm-guardrail.md)). |
| `mock-upstream` | A fake OpenAI-compatible model for development and the e2e suite. Port `127.0.0.1:8001`. Never run it anywhere real. |
| `bootstrap` | Runs once: creates teams `team-standard` and `team-strict`, users `alice` and `bob`, and one key for each user (restricted to the chat routes, see [Keys](#keys)) through LiteLLM's API, then exits. |
| `ollama` | Optional (`--profile ollama`): a small local model instead of the mock. |

Policies: `team-standard` gets `policies/default.yaml` and `team-strict` gets `policies/strict.yaml`, through the dev identity map in [`policies/identity.yaml`](policies/identity.yaml). The policy comes only from the identity LiteLLM verified from the key. Nothing in a request can choose it or turn the guardrail off.

## Quickstart

Run these from `deploy/compose/`:

```bash
cp .env.example .env
# Replace each value in .env, for example with: openssl rand -hex 32
# (LITELLM_MASTER_KEY, LLMFW_DEV_ALICE_KEY and LLMFW_DEV_BOB_KEY must start with "sk-")

docker compose up -d --build --wait
docker compose logs bootstrap        # "bootstrap done"
```

The first build takes a few minutes: it installs torch, spaCy and Presidio and downloads the sentence model.

Call the proxy with the openai SDK, using the key bootstrap created for alice:

```python
import os

from openai import OpenAI

# After: set -a && . ./.env && set +a
client = OpenAI(
    base_url="http://localhost:4000/v1", api_key=os.environ["LLMFW_DEV_ALICE_KEY"]
)
reply = client.chat.completions.create(
    model="chat", messages=[{"role": "user", "content": "scenario-pass"}]
)
print(reply.choices[0].message.content)
```

Mock scenarios (put the word in the user message):
- `scenario-pass`: a harmless answer.
- `scenario-leak-key`: an answer containing a made-up API key. Non-streaming calls get a 400. Streaming calls end with `finish_reason: "content_filter"`, and no part of the key is sent.
- `scenario-long`: a long harmless answer.

Audit events are JSON lines in the proxy log: `docker compose logs litellm | grep '"event_id"'`. They hold types, offsets, counts, timings, the team id and an HMAC of the user id, never text.

## Keys

**Every key must be created with `allowed_routes`.** LiteLLM treats a key with no `allowed_routes` (or an empty list) as allowed on every route, including `/v1/messages`, `/v1/responses` and its provider pass-through routes (`/anthropic/*`, `/gemini/*`, `/milvus/*`, ...). The guardrail therefore blocks any key whose `allowed_routes` is unrestricted, with `403 "key has no allowed_routes"`. That includes the master key: use it for management calls only.

bootstrap creates alice's and bob's keys with:

```json
"allowed_routes": [
  "/v1/chat/completions",
  "/chat/completions",
  "/engines/chat/chat/completions",
  "/openai/deployments/chat/chat/completions"
]
```

LiteLLM 1.108.3 matches an explicit entry by exact path or path prefix (`RouteChecks._route_matches_allowed_route`), and does not expand `{model}` placeholders in explicit entries. So the two deployment-style aliases are listed with this stack's model name, `chat`. `/cursor/chat/completions` is deliberately left out: other `/cursor/*` paths are LiteLLM's Cursor Cloud Agents pass-through, and it handles some request bodies through the Responses pipeline.

To create another key:

```bash
set -a && . ./.env && set +a
curl -s localhost:4000/key/generate \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "alice", "team_id": "team-standard",
       "allowed_routes": ["/v1/chat/completions", "/chat/completions"]}'
```

## e2e tests

From the repository root, with the stack up **including the test-only override** [`docker-compose.e2e.yml`](docker-compose.e2e.yml). It adds fake Anthropic and OpenAI credentials that point at the mock, so the pass-through routes are configured and the tests can prove they are still blocked:

```bash
cd deploy/compose
docker compose -f docker-compose.yml -f docker-compose.e2e.yml up -d --build --wait
cd ../..
set -a && . deploy/compose/.env && set +a
LLMFW_E2E_BASE_URL=http://127.0.0.1:4000 uv run pytest tests/e2e
```

Do not use the override for anything else. CI runs the same steps in the `e2e` job (`.github/workflows/ci.yml`).

## Using Ollama instead of the mock

In `.env`:

```bash
UPSTREAM_MODEL=ollama_chat/qwen2.5:0.5b
UPSTREAM_API_BASE=http://ollama:11434
OLLAMA_MODEL=qwen2.5:0.5b
```

```bash
docker compose --profile ollama up -d --build --wait
docker compose --profile ollama logs ollama-pull   # wait until the model is pulled
```

The e2e suite expects the mock: run it with the default upstream.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `LITELLM_MASTER_KEY` | required | LiteLLM admin key. |
| `POSTGRES_PASSWORD` | required | Password for the compose Postgres. |
| `LLMFW_AUDIT_HMAC_KEY` | required | Key for the HMAC-SHA256 of user ids in audit events. The guardrail will not start without it. |
| `LLMFW_DEV_ALICE_KEY`, `LLMFW_DEV_BOB_KEY` | required | The keys bootstrap creates for alice and bob. |
| `UPSTREAM_MODEL`, `UPSTREAM_API_BASE`, `UPSTREAM_API_KEY` | the mock | The model behind the `chat` model name. |
| `LLMFW_STREAM_HOLDBACK_CHARS` / `_CONTEXT_CHARS` / `_SCAN_STEP_CHARS` | 128 / 128 / 32 | The streaming window ([ADR 0004](../../docs/decisions/0004-streaming.md)). Not final: the latency task chooses N. |

## Notes

- Config files are copied into the images, not bind-mounted, so the stack also runs where Docker cannot read the checkout (for example a macOS folder Docker Desktop has no access to). Rebuild after editing them: `docker compose up -d --build`.
- LiteLLM's own logging is set explicitly to keep prompt and response text out: `LITELLM_LOG=WARNING` (docker-compose.yml), and `turn_off_message_logging`, `redact_messages_in_exceptions`, `log_raw_request_response`, `set_verbose` and `store_prompts_in_spend_logs` in `litellm/config.yaml`, each commented with where it is read in the litellm 1.108.3 source.
- `docker compose down -v` removes the database volume, and with it all keys, users and teams.
