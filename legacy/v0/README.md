# legacy/v0: the original app, frozen

This directory holds the original v0 app exactly as it was at commit `c14a773`. The only change is that it was moved here.

| Path | What it is |
|---|---|
| `app/` | FastAPI service (`main.py`), Streamlit UI (`ui.py`), LangGraph graph, v0 security modules |
| `config.py` | v0 roles, thresholds and jailbreak phrase list |
| `scripts/` | AstraDB ingestion (`ingest_data.py`, `ingest_enron.py`) and the old evaluators |
| `training/` | dataset preparation, SageMaker launcher, fine-tuning entry point and its requirements |
| `tests/test_firewall.py` | the original golden-dataset test |
| `docker-compose.yml` | the v0 backend and Streamlit services |
| `company_policy.txt`, `secrets_catalog.json` | v0 data files (the catalog is a snapshot; the engine reads the one at the repo root) |

## Frozen means

- It is not run, not tested in CI, not linted or type-checked, and not copied into the Docker image.
- It is not fixed. Known bugs are recorded in [docs/v0-findings.md](../../docs/v0-findings.md), whose links point to this code at commit `9311a98` on GitHub.
- It is not updated to match the rest of the repo. Its imports (`from app...`, `from config...`) assume `legacy/v0/` is the working directory, and it needs AstraDB and Ollama to run.

## Why

The v1 firewall is a guardrail plugged into LiteLLM, not a RAG app of its own. The v0 app has problems the v1 design removes rather than patches: a caller-chosen role, raw text in the audit log, and conversation state leaking between turns. Its detection code (input rules, PII redaction, output filter) was moved, with the same behavior, into `src/llmfirewall/engine/`. Characterization tests in `tests/` pin that behavior. The app is kept so the "what changed" story can be told against real code. The full reasoning is in [docs/decisions/0001-retire-v0-app.md](../../docs/decisions/0001-retire-v0-app.md).
