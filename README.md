# AI Security Gateway

A multi-layered security framework for AI agents that demonstrates defense-in-depth principles for RAG systems. Built to showcase practical implementation of prompt injection defense, PII redaction, RBAC-filtered retrieval, and semantic leak detection.

**Goal:** Provide a controlled AI assistant for enterprise use that prevents sensitive data leakage while maintaining usability.

## Architecture

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       ↓
┌─────────────────────────┐
│  Input Security Layer   │  • PII Redaction (Presidio)
│  (validation + redact)  │  • Prompt Injection Detection
└──────┬──────────────────┘  • Topic Classification
       ↓
┌─────────────────────────┐
│   Access Control Layer  │  • RBAC Permission Check
│   (RBAC filtering)      │  • User Role Validation
└──────┬──────────────────┘
       ↓
┌─────────────────────────┐
│   RAG Retrieval Layer   │  • Metadata-Filtered Search
│   (AstraDB vector DB)   │  • Access Level Enforcement
└──────┬──────────────────┘
       ↓
┌─────────────────────────┐
│   LLM Generation Layer  │  • Context-Grounded Response
│   (Ollama/Llama 3.2)    │  • Iterative Rewriting
└──────┬──────────────────┘
       ↓
┌─────────────────────────┐
│  Output Security Layer  │  • Ensemble Leak Detection:
│  (leak detection)       │    - Pattern Matching (regex)
└──────┬──────────────────┘    - Keyword Analysis
       ↓                        - Semantic Similarity
┌─────────────────────────┐
│   Audit & Logging       │  • Full Interaction Logs
│   (compliance trail)    │  • Security Event Tracking
└─────────────────────────┘
```

## Test Results

Evaluated against 160+ adversarial test cases spanning prompt injection, 
PII leakage, jailbreak attempts, and role-based access violations:

| Metric | Score | Details |
|--------|-------|---------|
| **Overall Security Score** | 86% | Across all 160+ test cases |
| **PII Redaction** | 100% | All SSN, emails, credit cards masked |
| **Input Security** | 100% | All prompt injections blocked (0% FNR) |
| **Jailbreak Defense** | 72% | Blocked across adversarial prompt variants |
| **Confidential Leak Reduction** | 92% | Via ensemble output validation |
| **Leak Detection** | 96% | Semantic + pattern matching ensemble |
| **RBAC Enforcement** | Passed | Metadata-filtered vector search working |
| **Response Time** | 8-10s | Bottleneck: Local Ollama inference |

### Test Case Breakdown
- Prompt injection variants: direct, indirect, Unicode obfuscation
- PII leakage attempts: SSN, email, credit card, API key extraction
- Jailbreak attempts: role-play, instruction override, context manipulation
- RBAC violations: cross-role document access attempts
- Synthetic dataset, manually crafted; informed by OWASP LLM Top 10

## Security Layers

### 1. Input Security
- **PII Redaction:** Microsoft Presidio detects and masks SSN, emails, credit cards, phone numbers, API keys
- **Prompt Injection Defense:** 27 jailbreak patterns + Unicode normalization to detect obfuscation
- **Topic Classification:** Cosine similarity determines query sensitivity level

### 2. Access Control (RBAC)
- **Role Permissions:**
  - Employees: `public` + `internal` documents
  - Managers: + `confidential` documents
  - Executives: + `secret` documents
- **Implementation:** AstraDB metadata filtering on `access_level` field

### 3. Output Validation (Ensemble Approach)
- **Layer 1:** Regex patterns for structured secrets (passcodes, API keys, salary figures)
- **Layer 2:** Keyword detection (salary, password, budget, confidential)
- **Layer 3:** Semantic similarity against secrets catalog (threshold: 0.75)
- **Decision Logic:** Block if any layer triggers + similarity > 0.65

### 4. Audit & Compliance
- All interactions logged to AstraDB with user attribution, timestamps, and security decisions

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Agent Framework** | LangGraph | State machine with conditional routing |
| **LLM** | Ollama (Llama 3.2 1B) | Local inference, cost-free |
| **Vector DB** | DataStax AstraDB | Knowledge base + audit logs |
| **PII Detection** | Microsoft Presidio | Named entity recognition |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) | Semantic similarity |
| **Backend** | FastAPI | REST API with /chat and /health endpoints |
| **Frontend** | Streamlit | Interactive demo UI |
| **Deployment** | Docker Compose | Containerized services |

## Performance Analysis

**Current Bottleneck:** Ollama local inference (8-10s per request)

**Optimization Paths:**
1. **Model swap:** llama3.2:1b (current) → faster distilled models
2. **Response streaming:** Server-Sent Events for perceived speed
3. **Query caching:** Redis for identical/similar queries
4. **Parallel execution:** Run RAG + PII redaction concurrently

## Metrics & Evaluation

This project includes comprehensive metrics tracking and evaluation:

- **Golden Dataset Testing**: Automated validation against labeled test cases
- **Performance Benchmarking**: Latency, throughput, and resource usage measurement
- **Live Metrics API**: Real-time monitoring of security and performance KPIs

**Run Evaluation:**
```bash
python scripts/evaluate_system.py
```

**Run Benchmark:**
```bash
python scripts/benchmark.py
```

**View Live Metrics:**
```bash
curl http://localhost:8000/metrics
```

See [METRICS.md](METRICS.md) for detailed metrics documentation and resume-worthy talking points.

---

**Note**: This is a portfolio project demonstrating AI security concepts. It provides the **technical layer** of an AI governance strategy. Complete enterprise deployment would also require policy enforcement, user training, and network-level access controls.
