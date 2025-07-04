# Sekai Recommendation Agent

> **AI-Native & Cutting-Edge Recommendation System for Role-Play Stories**

A sophisticated recommendation system that leverages the latest LLMs and AI frameworks to provide personalized story recommendations for Sekai users. This project demonstrates deep content understanding, advanced prompt engineering, and robust evaluation methodologies.

## Directory Structure

- `src/agents/` — Core agent implementations
- `src/models/` — Model wrappers and utilities
- `data/` — Raw and processed data, embeddings, FAISS index
- `logs/` — Experiment logs, evaluation history, prompt evolution
- `configs/` — Config files (if any)
- `tests/` — Unit and integration tests

---

## Quick Results

| Cycle | P@10      | R@10      | Δ vs prev |
| ----- | --------- | --------- | --------- |
| 0     | 0.400     | 0.400     | —         |
| 1     | 0.540     | 0.540     | +0.140    |
| 2     | **0.760** | **0.760** | +0.220    |

> **Best Prompt**: v2.0 (Cycle 2) – +90% precision gain.

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

## Architecture & agent roles

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Multi-View      │    │ Recommendation   │    │ Evaluation      │
│ Recall System   │───▶│ Agent            │───▶│ Agent           │
│ (Tag+Semantic+  │    │ (LLM Rerank)     │    │ (LLM Eval)      │
│  Rule+Popular)  │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Prompt Optimizer │
                       │ Agent            │
                       └──────────────────┘
```

The system is composed of three main agents, orchestrated in an evaluation-optimization loop:

- **RecommendationAgent**:

  - Receives user interest tags and returns the top-K most relevant content IDs.
  - Uses a hybrid approach: fast vector recall (FAISS + embeddings) and LLM-based reranking.
  - Supports prompt hot-swapping and template optimization.

- **EvaluationAgent**:

  - Evaluates the quality of recommendations using either LLM-based or keyword-based ground truth.
  - Simulates user tag selection and computes metrics (precision, recall, F1).
  - Logs detailed evaluation results for each user and cycle.

- **PromptOptimizerAgent**:
  - Analyzes evaluation history and dynamically updates the system prompt for the RecommendationAgent.
  - Supports both exploit (incremental improvement) and explore (diversification) strategies.
  - Triggers prompt updates based on observed metric gains or stagnation.

The **Orchestrator** coordinates these agents, running multi-cycle experiments, logging results, and managing prompt evolution.

## 📊 Performance & Robustness

### Current Performance

- **Precision@10**: ~0.49 (baseline)
- **Recall@10**: ~0.49 (baseline)
- **Multi-View Recall**: 500+ candidates from diverse sources
- **LLM Rerank**: 30-candidate window for quality optimization

### Robustness Features

- **Fallback Mechanisms**: Vector recall when LLM unavailable
- **Error Handling**: Graceful degradation on API failures
- **Cost Monitoring**: Real-time token usage tracking
- **Cross-Validation**: Multiple evaluation modes (LLM + keyword)

## 📈 Evaluation & Analysis

### Comprehensive Metrics

- **Precision@K**: Accuracy of top-K recommendations
- **Recall@K**: Coverage of relevant content
- **Method Distribution**: Analysis of recall sources
- **Cost Tracking**: Token usage and API costs

### Prompt Strategy Templates

See `prompt_strategy_templates.md` for advanced prompt engineering techniques:

- LLM rerank strategies
- Cold-start optimization
- Tag upweighting methods
- Diversity enhancement

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:

- `openai` / `google-generativeai`: LLM APIs
- `sentence-transformers`: Embedding models
- `whoosh`: Inverted indexing
- `faiss`: Vector similarity search
- `pandas` / `numpy`: Data processing

## 📄 License

MIT License - see LICENSE file for details.

---

**Built with ❤️ for the Sekai community**

---

## Caching Strategy

- **Embedding Cache**:

  - Content embeddings are precomputed and stored (e.g., in `data/contents_embeddings.npy` and `faiss_index.bin`).
  - FAISS is used for efficient vector search and recall.

- **Prompt/Inference Cache**:
  - Agents inherit from a `CachedAgent` base class, which provides in-memory caching for expensive model calls (e.g., LLM generations).
  - Caches are keyed by stable, serialized input representations.
  - Cache statistics (hit/miss rates) are tracked for performance monitoring.

---

## Evaluation Metric & Stopping Rule

- **Metrics**:

  - The system computes `precision@k`, `recall@k`, and `f1@k` for each recommendation batch.
  - Metrics are logged per user and aggregated across cycles.

- **Stopping Rule**:
  - The prompt optimizer monitors the improvement (`delta`) in metrics across cycles.
  - If the expected gain in precision/recall falls below a configurable threshold (`min_delta`) for multiple rounds, the optimizer triggers an "explore" strategy or halts further prompt updates.
  - This ensures efficient convergence and avoids overfitting to noise.
