# Baseline (v0, before any rework)

Recorded 2026-11-19 on branch `v1/01-baseline`, created from `origin/main` at `215f8dd`. No application code was changed. This is the "before" picture: whether the existing code installs and passes its tests, and how fast its three checks are.

## Environment

| | |
|---|---|
| OS | macOS 15.7.9 (build 24G830), arm64 |
| CPU | Apple M3, 8 cores (4 performance + 4 efficiency) |
| RAM | 24 GB |
| Python | 3.12.16 (installed with uv 0.13.9), venv at `.venv/` |
| Torch device | CPU only. MPS is available on this machine but was not used, and CUDA was hidden. |
| Torch threads | 4 (`torch.get_num_threads()`, default) |

Key package versions. `requirements.txt` is unpinned, so these are just what resolved on this date. The full freeze is in the appendix.

| Package | Version |
|---|---|
| presidio-analyzer / presidio-anonymizer | 2.2.366 |
| spacy | 3.8.18 |
| en-core-web-lg (spaCy model) | 3.8.0 |
| sentence-transformers | 6.3.0 |
| all-MiniLM-L6-v2 | from the Hugging Face cache |
| transformers | 5.23.0 |
| torch | 2.15.0 |
| numpy | 2.4.6 |
| langchain | 1.5.2 |
| langgraph | 1.2.23 |
| fastapi | 0.165.0 |
| pytest | 9.2.0 |

Install: every package in `requirements.txt` installed, both models downloaded, and nothing failed.

## Test results

Command: `.venv/bin/pytest tests/ -v`

```
platform darwin -- Python 3.12.16, pytest-9.2.0, pluggy-1.7.0
collected 45 items
...
============================= 45 passed in 20.02s ==============================
```

**45 passed, 0 failed, 0 skipped. That overstates what is tested.** All 45 cases come from `tests/data/golden_dataset.json` through one parametrized test, `test_firewall_logic`:

| `expected_type` | Cases | Case indices | What the test asserts |
|---|---|---|---|
| `redaction` | 9 | 0–3, 15, 32–34, 44 | `input_security_node` redacts `should_not_contain` |
| `audit_fail` | 5 | 4–6, 26, 27 | `output_security_node` blocks |
| `pass` | 15 | 7–9, 13, 14, 16–18, 28–31, 40–42 | `output_security_node` does not block |
| `input_block` | **16** | 10–12, 19–25, 35–39, 43 | **Nothing.** The test has no branch for `input_block`, so these cases reach the end with no assertion and are reported as PASSED. |

So 29 of the 45 passes check anything, and the prompt-injection check (`validate_input`) is never asserted by any test.

Other notes:
- The suite is not offline. Importing `app.graph.nodes` loads the Presidio/spaCy and MiniLM models, and it downloads them if they are not cached.
- pytest-asyncio prints a `PytestDeprecationWarning` because `asyncio_default_fixture_loop_scope` is unset. The warning does not affect results.

## Latency ("before")

Script: [`bench/baseline_v0.py`](../bench/baseline_v0.py), run as `.venv/bin/python bench/baseline_v0.py`.

It times each function directly, not the LangGraph nodes:

| Component | Function called |
|---|---|
| input check | `app.security.input_validator.validate_input(text, "employee")` |
| PII redaction | `app.redactor.redact_pii(text)` (Presidio + spaCy `en_core_web_lg`) |
| output filter | `app.security.output_filter.validate_output(text)` (regex, keywords, MiniLM similarity against `secrets_catalog.json`) |

Inputs: the `input` field of every case in `tests/data/golden_dataset.json`, **n = 45** (the dataset has 45 cases, not 50).

**The output filter was timed on golden user inputs, not on model answers.** Real answers are usually longer, which would make its timings higher.

How the CPU-only setting was applied: the app constructs `SentenceTransformer('all-MiniLM-L6-v2')` without a device, which would select MPS on Apple silicon. Before `output_filter` is imported, the benchmark wraps that constructor so it passes `device="cpu"`. It then asserts `model.device == cpu`. spaCy runs on CPU by default. No app code is modified. Hugging Face runs in offline mode (`HF_HUB_OFFLINE=1`), so load times are cache reads, not network calls.

### Model loading excluded (warm)

One process. The script imports each component (models load during import), makes one warm-up call per component and discards it, then times 5 passes over the 45 inputs, giving 225 timings per component. It uses `time.perf_counter`, and percentiles are linearly interpolated.

| Component | n | p50 (ms) | p95 (ms) | mean (ms) | min (ms) | max (ms) |
|---|---|---|---|---|---|---|
| input check | 225 | 0.003 | 0.004 | 0.003 | 0.000 | 0.039 |
| PII redaction | 225 | 3.292 | 4.768 | 3.564 | 2.435 | 23.741 |
| output filter | 225 | 212.922 | 269.226 | 189.417 | 0.002 | 1068.430 |

### Model loading included (cold)

10 fresh Python processes per component. Each one times importing the component (which loads its models) plus the first call on one input. Run *i* uses input *i*. Python interpreter startup is not included.

| Component | n | p50 (ms) | p95 (ms) | mean (ms) | min (ms) | max (ms) |
|---|---|---|---|---|---|---|
| input check | 10 | 15.917 | 20.844 | 16.771 | 15.564 | 24.684 |
| PII redaction | 10 | 4937.445 | 5571.165 | 5001.982 | 4660.145 | 5690.237 |
| output filter | 10 | 4425.977 | 4733.645 | 4463.584 | 4135.433 | 4755.586 |

### Reading the numbers

- **The output filter dominates.** Its warm p50 is about 213 ms, roughly 65 times the PII redaction p50. `check_semantic_similarity` re-encodes all 27 entries of `secrets_catalog.json` with MiniLM on every call, instead of encoding them once. That is 28 encoder passes per call. This is recorded here, not fixed.
- **The output filter's timings fall into two groups.** 8 of the 45 inputs match a regex in `check_pattern_match` and return before any embedding, which explains the 0.002 ms minimum and why the mean is below the p50. The other 37 take the full embedding path.
- **The 1068 ms output-filter maximum is a single outlier.** I did not investigate its cause. The p95 (269 ms) is the representative upper figure.
- **The input check is pure Python string matching**, a few microseconds per call. Its cold cost is mostly importing `config`.
- **Cold start is about 4–5 s each for PII redaction and the output filter**, almost all of it model loading (spaCy `en_core_web_lg`, and torch plus MiniLM). A process that loads both pays roughly the sum, because these benchmarks measured them in separate processes.
- One run on one machine. Treat these numbers as a reference point, not a benchmark result with error bars.

## Appendix: `uv pip freeze` (.venv, 2026-11-19)

```
aiohappyeyeballs==2.7.1
aiohttp==3.14.5
aiosignal==1.4.0
altair==6.3.0
annotated-doc==0.0.7
annotated-types==0.8.0
anyio==4.16.0
astrapy==2.3.2
asttokens==3.0.2
attrs==26.1.0
blis==1.3.4
cassandra-asyncio-driver==0.1.4
cassandra-driver==3.30.1
cassio==0.1.10
catalogue==2.0.10
certifi==2026.10.23
cffi==2.1.1
charset-normalizer==3.5.2
click==8.5.1
cloudpathlib==0.26.0
cloudpickle==3.1.3
confection==1.3.4
cryptography==50.0.2
cymem==2.0.16
deprecated==3.0.1
deprecation==2.1.0
distro==1.9.0
dnspython==2.9.2
en-core-web-lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
executing==2.3.0
fastapi==0.165.0
filelock==4.2.12
frozenlist==1.8.0
fsspec==2026.9.0
geomet==1.1.0
h11==0.16.0
h2==4.4.1
hf-xet==1.8.0
hpack==4.2.0
httpcore==1.0.9
httpcore2==2.16.2
httpx==0.28.1
httpx-sse==0.4.3
httpx2==2.16.2
huggingface-hub==1.33.0
hyperframe==6.1.0
idna==3.20
iniconfig==2.3.1
ipython==9.18.1
ipython-pygments-lexers==1.1.1
itsdangerous==2.2.0
jedi==0.20.1
jinja2==3.1.6
joblib==1.6.0
jsonpatch==1.47
jsonpointer==3.12
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
langchain==1.5.2
langchain-astradb==1.0.0
langchain-classic==1.0.8
langchain-community==0.4.2
langchain-core==1.8.3
langchain-huggingface==1.2.3
langchain-ollama==1.1.0
langchain-protocol==0.0.21
langchain-text-splitters==1.1.3
langgraph==1.2.23
langgraph-checkpoint==4.3.2
langgraph-checkpoint-cassandra==0.1.1
langgraph-prebuilt==1.1.3
langgraph-sdk==0.4.14
langsmith==0.15.0
markdown-it-py==4.3.0
markupsafe==3.0.4
matplotlib-inline==0.2.2
mdurl==0.1.2
mpmath==1.3.0
multidict==7.2.1
murmurhash==1.0.16
narwhals==2.29.0
networkx==3.7
numpy==2.4.6
ollama==0.6.3
opentelemetry-api==1.45.2
orjson==3.13.0
ormsgpack==1.13.1
packaging==26.3
pandas==3.1.1
parso==0.8.7
pexpect==4.9.0
phonenumbers==9.0.43
pillow==13.0.0
pluggy==1.7.0
preshed==3.0.13
presidio-analyzer==2.2.366
presidio-anonymizer==2.2.366
prompt-toolkit==3.0.53
propcache==0.5.4
protobuf==7.36.2
psutil==7.2.2
ptyprocess==0.7.0
pure-eval==0.2.4
pyarrow==26.0.1
pycparser==3.12
pydantic==2.14.3
pydantic-core==2.50.4
pydantic-settings==2.16.3
pydeck==0.9.3
pygments==2.21.0
pymongo==4.20.0
pytest==9.2.0
pytest-asyncio==1.5.0
python-dateutil==4.0.0
python-dotenv==1.2.4
python-multipart==0.0.34
pyyaml==6.0.3
referencing==0.37.0
regex==2026.11.19
requests==2.34.2
requests-file==3.0.1
requests-toolbelt==1.0.0
rich==15.0.0
rpds-py==2026.9.1
safetensors==0.8.0
scikit-learn==1.9.1
scipy==1.18.1
sentence-transformers==6.3.0
setuptools==84.2.1
shellingham==1.5.4
smart-open==9.0.5
sniffio==1.3.1
spacy==3.8.18
spacy-legacy==3.0.12
spacy-loggers==1.0.5
sqlalchemy==2.1.9
srsly==2.5.6
stack-data==0.6.3
starlette==1.10.0
streamlit==1.67.0
sympy==1.14.0
tenacity==9.2.1
thinc==8.3.14
threadpoolctl==3.8.0
tldextract==5.4.0
tokenizers==0.23.3
toml==0.10.2
torch==2.15.0
tqdm==4.70.1
traitlets==5.16.1
transformers==5.23.0
truststore==0.10.5
typer==0.29.2
typing-extensions==4.16.0
typing-inspection==0.4.5
tzdata==2026.6
urllib3==2.8.0
uuid-utils==0.19.0
uuid6==2025.0.1
uvicorn==0.54.0
wasabi==1.1.3
wcwidth==0.9.4
weasel==1.0.1
websockets==17.4
wrapt==2.5.2
xxhash==4.1.0
yarl==1.25.1
zstandard==0.25.0
```
