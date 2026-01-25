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

From evaluation with 50-case golden dataset:

| Metric | Score | Details |
|--------|-------|---------|
| **PII Redaction** | 100% | All SSN, emails, credit cards masked |
| **Input Security** | 100% | All prompt injections blocked (0% FNR) |
| **Leak Detection** | 96% | Semantic + pattern matching ensemble |
| **RBAC Enforcement** | ✅ | Metadata-filtered vector search working |
| **Response Time** | 8-10s | Bottleneck: Local Ollama inference |

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

**Production Alternative:** Gemini API (5-7s) but requires API key + costs

## Limitations

**Acknowledged Trade-offs:**
- ❌ No real users → validation is synthetic (50 test cases)
- ❌ Fixed thresholds → no learning from false positives
- ❌ Circular validation → same person wrote attacks and defenses
- ❌ Metadata filtering → relies on proper document tagging at ingestion
- ❌ Local LLM → slow inference, limited reasoning compared to cloud models

**Production Gaps:**
- No JWT authentication (demo uses role parameter)
- No rate limiting or DDoS protection
- No health monitoring/alerting integration
- Conversation history lost on restart (MemorySaver, not Cassandra persistence)

## What This Demonstrates

**For Entry-Level AI/ML Engineering Roles:**

✅ **System Design:** Multi-layer security architecture, not just API calls  
✅ **Agent Frameworks:** LangGraph state machines with conditional routing  
✅ **RAG Engineering:** Metadata-filtered vector search with access control  
✅ **Security Awareness:** Understanding of prompt injection, PII leakage, RBAC  
✅ **Testing Rigor:** 50 adversarial test cases with quantifiable metrics  
✅ **Trade-off Analysis:** Can articulate limitations and optimization paths  

**Not Claiming:**
- Battle-tested production system
- Novel ML techniques (using existing tools correctly)
- Scale validation (no load testing)
- Real-world user validation

## Limitations & Production Considerations

This is a **proof-of-concept demonstrating AI security techniques**. Production deployment would require:

- **Authentication/Authorization**: OAuth2, SAML, or enterprise SSO integration
- **Network-Level Controls**: Firewall rules to enforce gateway usage
- **Advanced Threat Detection**: ML-based anomaly detection for behavioral analysis
- **Regulatory Compliance**: GDPR, HIPAA, SOC2 audit trail enhancements
- **High Availability**: Multi-region deployment with failover
- **Performance Optimization**: Caching, load balancing, and latency reduction
- **Real-Time Alerting**: Integration with SIEM systems for security incidents

## What This Demonstrates

For recruiters and technical reviewers, this project showcases:

1. **System Design Skills**: Multi-layer security architecture with proper separation of concerns
2. **AI Safety Understanding**: Awareness of prompt injection, data leaks, and access control
3. **Modern Tech Proficiency**: LangGraph agents, RAG systems, vector databases
4. **Security-First Thinking**: Input validation before processing, not just output filtering
5. **Production Awareness**: Clear documentation of limitations and production requirements

## Installation & Setup

See deployment instructions in the original README sections for Docker Compose setup and environment configuration.

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
