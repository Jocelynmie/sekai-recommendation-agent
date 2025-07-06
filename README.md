# ❤️ Sekai Recommendation Agent

> **AI-Native & Cutting-Edge Recommendation System for Role-Play Stories**

A sophisticated recommendation system that leverages the latest LLMs and AI frameworks to provide personalized story recommendations for Sekai users. This project demonstrates deep content understanding, advanced prompt engineering, and robust evaluation methodologies.


## 📖 Directory Structure

- `src/agents/` — Core agent implementations
- `src/models/` — Model wrappers and utilities
- `data/` — Raw and processed data, embeddings, FAISS index
- `logs/` — Experiment logs, evaluation history, prompt evolution
- `configs/` — Config files (if any)
- `tests/` — Unit and integration tests

---

## 🍎 Quick Results

| Cycle | P@10      | R@10      | Δ vs prev |
| ----- | --------- | --------- | --------- |
| 0     | 0.400     | 0.400     | —         |
| 1     | 0.540     | 0.540     | +0.140    |
| 2     | **0.760** | **0.760** | +0.220    |

> **Best Prompt**: v2.0 (Cycle 2) – +90% precision gain.

---

## 🚀 Quick Start

**One command to run a complete experiment:**

```bash
# Quick test (1 cycle, 5 users)
python run_experiment.py --cycles 1 --users 5

# Full experiment (3 cycles, 15 users, LLM mode)
python run_experiment.py --cycles 3 --users 15 --mode llm

# Vector mode (faster, lower cost)
python run_experiment.py --cycles 2 --users 10 --mode vector
```

### 📋 Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:

- **LLM APIs**: `google-generativeai` / `openai` / `anthropic`
- **向量和搜索**: `sentence-transformers` / `faiss-cpu` / `whoosh`
- **数据处理**: `pandas` / `numpy`
- **日志和配置**: `loguru` / `python-dotenv`
- **工具类**: `tqdm`

---

## 🏠 Architecture & Agent Roles

```
┌──────────┐
│  Client  │
└───▲──┬───┘
    │  │behavior
    │  │logs
    ▼  │
┌────────────────────────┐
│   A/B Router & Guard   │◀── BudgetMonitor ──┐
└───────┬────────┬───────┘                  │
        │        │                          │
        │latency │fallback                  │cost
        ▼        ▼                          │
┌──────────────┐  ┌─────────────────┐       │
│ Vector Store │  │ Multi‑View      │       │
│  (FAISS)     │  │  Recall System  │       │
└─────┬────────┘  └────────┬────────┘       │
      │ candidates         │                │
      ▼                    ▼                │
┌──────────────────┐   ┌──────────────────┐  │
│ Recommendation   │   │  Policy Guard    │──┘
│  Agent (LLM)     │   │  (safety/fair)   │
└────────┬─────────┘   └────────┬─────────┘
         │ ranked               │filtered
         ▼                      ▼
      ┌───────────────────────────────────┐
      │           User Feed               │
      └───────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │ Evaluation Agent │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │ Prompt Optimizer │
              └──────────────────┘
```

**System Overview:**

- **Client**: Sends user behavior and receives recommendations.
- **A/B Router & Guard**: Handles traffic splitting, latency fallback, and cost control (with BudgetMonitor).
- **Vector Store (FAISS)**: Stores content embeddings for fast candidate retrieval.
- **Multi-View Recall System**: Combines tag, semantic, rule, and popularity signals to generate diverse candidates.
- **Recommendation Agent (LLM)**: Reranks candidates using LLMs for high-quality recommendations.
- **Policy Guard**: Applies safety and fairness filtering to the ranked list.
- **User Feed**: Final recommendations delivered to the user.
- **Evaluation Agent**: Continuously evaluates recommendation quality and logs metrics.
- **Prompt Optimizer**: Dynamically updates prompts and strategies based on evaluation feedback.
- **BudgetMonitor**: Monitors and controls API usage and cost.

---

## 🔋 Caching Strategy

- **Embedding Cache**:

  - Content embeddings are precomputed and stored (e.g., in `data/contents_embeddings.npy` and `faiss_index.bin`).
  - FAISS is used for efficient vector search and recall.

- **Prompt/Inference Cache**:
  - Agents inherit from a `CachedAgent` base class, which provides in-memory caching for expensive model calls (e.g., LLM generations).
  - Caches are keyed by stable, serialized input representations.
  - Cache statistics (hit/miss rates) are tracked for performance monitoring.

---

## 🌟 Evaluation Metric & Stopping Rule

- **Metrics**:

  - The system computes `precision@k`, `recall@k`, and `f1@k` for each recommendation batch.
  - Metrics are logged per user and aggregated across cycles.

- **Stopping Rule**:
  - The prompt optimizer monitors the improvement (`delta`) in metrics across cycles.
  - If the expected gain in precision/recall falls below a configurable threshold (`min_delta`) for multiple rounds, the optimizer triggers an "explore" strategy or halts further prompt updates.
  - This ensures efficient convergence and avoids overfitting to noise.

---

## ☁️ Production Scaling Strategy

### Infrastructure Scaling
- **Horizontal Scaling**: Deploy agents as stateless microservices on Kubernetes
- **Vector Index**: Migrate from FAISS to Milvus/Qdrant for distributed vector search (currently using FAISS)
- **Caching Layer**: Implement Redis cluster for distributed prompt/embedding cache
- **Model Serving**: Use vLLM or TensorRT-LLM for optimized inference at scale

### Performance Optimization
- **Batch Processing**: Queue-based architecture for handling bulk recommendations
- **Async Pipeline**: Non-blocking agent communication via message queues (e.g., Kafka)
- **Load Balancing**: Implement request routing based on model availability and latency

### Monitoring & Observability
- **Metrics**: Prometheus + Grafana for real-time performance monitoring
- **Logging**: Centralized logging with ELK stack
- **Tracing**: Distributed tracing with OpenTelemetry

---

## 🤖 Model Selection Strategy

Our system uses a **cutting-edge, cost-effective model stack** with the latest AI models from multiple vendors:

### **Model Configuration**

| Agent                    | Primary Model     | Key Advantages                                   | Cost Savings     |
| ------------------------ | ----------------- | ------------------------------------------------ | ---------------- |
| **RecommendationAgent**  | Gemini 2.0 Flash  | Latest "no-thinking" model, >300 RPS, free quota | 100% (free tier) |
| **EvaluationAgent**      | Gemini 2.5 Pro    | June 2025 release, SOTA reasoning, 1.5×-2× speed | 20% vs GPT-4o    |
| **PromptOptimizerAgent** | Claude 3.5 Sonnet | Strong creativity/analysis, latest Claude series | 40% vs GPT-4o    |

### **Technical Benefits**

- **Latest Models**: Uses 2025 releases (Gemini 2.5 Pro, Claude 3.5 Sonnet)
- **Multi-Vendor**: Google + Anthropic + OpenAI fallbacks
- **Cost Optimization**: 30% total savings vs all-GPT-4o
- **Performance**: 1.5×-2× speed improvement

> 📖 **Detailed Analysis**: See [`MODEL_SELECTION_STRATEGY.md`](./MODEL_SELECTION_STRATEGY.md) for comprehensive model selection rationale and cost analysis.

---

## 📊 Performance & Robustness

### Baseline Metrics (Cycle 0)

| Environment        | Precision@10 | Recall@10 |
| ------------------ | ------------ | --------- |
| **run_free**       | **0.426**    | 0.438     |
| **run_experiment** | **0.573**    | 0.861     |

> _Values from `summary.json` auto-aggregation; different environments not directly comparable, for trend reference only._

- **Multi-View Recall**: ≈ 220 candidates (Tag 50 + Semantic 120 + Rule 20 + Hot 30).
- **LLM Re-rank Window**: Top-30 candidates are rescored by the Recommendation-LLM.

### Robustness Features

| Mechanism            | Description                                                                                           |
| -------------------- | ----------------------------------------------------------------------------------------------------- |
| **Fallback**         | `--mode vector` or auto-switch to vector ranking when API-Key missing.                                |
| **Error Handling**   | Each LLM call `max_retries=3` with exponential backoff; multiple failures throw `ModelError` and log. |
| **Cost Monitoring**  | `budget_monitor` aggregates tokens & \$ per cycle, output at log end.                                 |
| **Cross-Validation** | Supports `llm`, `keyword`, `vector` 3 evaluation modes for real click alignment. 
                     
---

## 📈 Evaluation & Analysis

### Comprehensive Metrics

- **Precision@K / Recall@K** – computed each cycle (`k` defaults to 10; override with `--k`).
- **Cost Tracking** – per‑cycle token & \$ summary via `budget_monitor`.
- **(Planned) Method‑Distribution** – future work: log source (tag / semantic / rule / hot) for each recalled candidate to analyse mix ratios.

### Prompt Strategy Templates

_For human readers / prompt engineers only._  
See [`prompt_strategy_templates.md`](./prompt_strategy_templates.md) for reference patterns:

- LLM re‑rank prompt used in `RecommendationAgent`.
- Cold‑start weighting & tag‑upweight logic reflected in `RecommendationAgent._score_candidates()`.
- Diversity enhancement and evaluation prompts mirror the templates, but **are embedded as Python string constants**, not imported from the Markdown file.

---

## 📄 License


This is a technical assessment project. For production use, please contact the Sekai team.

---

**Built with ❤️ for the Sekai community**

---
