# Agentic LLM Firewall

A security gateway for an enterprise RAG assistant. It screens what goes into the model, restricts what the model can retrieve to what the user is allowed to see, checks what comes out for leaked secrets, and logs every decision.

## Architecture

Two paths: an **online** path that answers each request, and an **offline** path that prepares data and trains an injection classifier.

### Online — one request

```
Streamlit UI ──► FastAPI POST /chat ──► LangGraph state machine
                                          │
   input_security ── blocked? ───────────►│
        │                                 │
   controlled_rag                         │
        │                                 │
   llm ◄── retry with rewrite instruction │
        │        (up to 3 retries)        │
   output_security ───────────────────────┤
                                          ▼
                                        audit ──► 200 answer / 403 reason
```

| Node | What it does | Code |
|---|---|---|
| `input_security` | Rejects empty, over-length, CSV-formula and symbol-heavy input; substring match against 27 jailbreak phrases (after NFKC Unicode normalisation); classifies the question's sensitivity by similarity to example phrases and checks the user's role may ask it; redacts PII with Microsoft Presidio | `app/security/input_validator.py`, `access_control.py`, `app/redactor.py` |
| `controlled_rag` | Embeds the redacted query (all-MiniLM-L6-v2) and retrieves the top 2 documents from AstraDB, **filtered inside the vector query** to the access levels the role allows | `app/graph/nodes.py` |
| `llm` | Answers from the retrieved context with a local Ollama model (`llama3.2:1b`, temperature 0.1) | `app/graph/nodes.py` |
| `output_security` | Checks the answer against regex patterns and semantic similarity to a catalog of known secrets; on failure, loops back to the LLM with a rewrite instruction, then blocks | `app/security/output_filter.py` |
| `audit` | Appends one JSON line per request to `security_audit.log` | `app/security/audit_logger.py` |

Roles map to document access levels in `config.py`: `employee` → public, internal · `manager` → + confidential · `executive` → + secret.

### Offline — data and model

| Step | Script |
|---|---|
| Load the demo policy documents into AstraDB | `scripts/ingest_data.py` |
| Load the Enron email corpus into AstraDB, with quality filtering, access levels derived from the sender's role, and resumable batched writes | `scripts/ingest_enron.py` |
| Build an injection-detection dataset (attacks: Microsoft LLMail-Inject; benign: deepset/prompt-injections + corpus text) and upload it to S3 | `scripts/prepare_finetune_data.py` |
| Fine-tune `protectai/deberta-v3-base-prompt-injection-v2` on SageMaker (`ml.g4dn.xlarge`), selecting the checkpoint by F2, tracked in Weights & Biases | `scripts/run_sagemaker_job.py` → `training/finetune_train.py` |
| Evaluate injection detection on external datasets — *`evaluate_injection_detection.py` does not currently run: it imports detector functions missing from `input_validator.py`. Both scripts are being replaced by a held-out evaluation harness.* | `scripts/evaluate_injection_detection.py`, `scripts/evaluate_jailbreakbench.py` |

## Known limitations

These are tracked, not hidden. Several are the subject of the current rebuild.

**Evaluation**
- The only committed test set (`tests/data/golden_dataset.json`, 45 cases) was written by the same author as the defenses, so it can't measure how they generalise.
- There is no end-to-end measurement yet of attack success against the whole system, of over-blocking of legitimate questions, or of latency.
- The training pipeline has known data-quality issues (benign and attack examples come from different sources, so a classifier can learn the source rather than the attack) and evaluates on data that overlaps its training set.

**Security**
- The user's role is supplied by the client in the request body. There is no authentication.
- Retrieved documents are not scanned for injected instructions before reaching the model, and they are placed in the system prompt.
- The audit log stores the original, unredacted query.
- The trained classifier is not yet used at request time.

**Correctness and performance**
- The keyword layer of the output filter can never trigger, and the dollar-amount pattern also blocks legitimate policy text (e.g. "$500").
- The `/chat` handler is `async` but runs blocking work, so it doesn't serve requests concurrently.
- Conversation checkpoints are kept in memory (`MemorySaver`) and lost on restart; Cassandra checkpointing is disabled pending a serialization fix.
- Documents are not chunked; long emails are truncated at 1500 words, beyond what the embedding model reads (256 tokens).

## Running it

**Prerequisites:** Docker, an AstraDB vector database, and [Ollama](https://ollama.com) running on the host with the model pulled:

```bash
ollama pull llama3.2:1b
```

**Configure:**

```bash
cp .env.example .env      # then fill in the AstraDB values
```

**Load the demo documents** (needs `pip install -r requirements.txt`):

```bash
python scripts/ingest_data.py
```

**Start the API and UI:**

```bash
docker compose up --build
# API: http://localhost:8000   (GET /health, GET /metrics, POST /chat)
# UI:  http://localhost:8501
```

**Tests:**

```bash
python -m spacy download en_core_web_lg   # Presidio's NER model, needed at import time
pytest tests/
```

The offline pipeline needs `pip install -r requirements-train.txt` plus the AWS and W&B values in `.env`. Run the scripts from the repository root.

## Repository layout

```
app/                FastAPI service, Streamlit UI, LangGraph graph, security layers
scripts/            ingestion, dataset preparation, SageMaker launcher, evaluators
training/           SageMaker training entry point and its container requirements
tests/              pytest suite and golden dataset
config.py           roles, thresholds, jailbreak phrase list
secrets_catalog.json  known secrets used by the output filter
```

---

This is a portfolio project. It covers the technical layer of AI governance; a real deployment would also need authentication, policy, and network-level controls.
