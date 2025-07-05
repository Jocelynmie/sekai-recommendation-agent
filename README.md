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

### 📋 Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:

- `openai` / `google-generativeai`: LLM APIs
- `sentence-transformers`: Embedding models
- `whoosh`: Inverted indexing
- `faiss`: Vector similarity search
- `pandas` / `numpy`: Data processing

## Architecture & Agent Roles

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

### **RecommendationAgent**

- **Primary Model**: Gemini 2.0 Flash (latest "no-thinking" model, >300 RPS, free quota)
- **Fallback**: GPT-4o Mini
- **Function**: Receives user interest tags and returns the top-K most relevant content IDs
- **Approach**: Hybrid vector recall (FAISS + embeddings) + LLM-based reranking
- **Features**: Prompt hot-swapping and template optimization

### **EvaluationAgent**

- **Primary Model**: Gemini 2.5 Pro (June 2025 release, SOTA reasoning, 20% cheaper than GPT-4o)
- **Fallback**: GPT-4o, GPT-4, Claude 3.5 Haiku
- **Function**: Evaluates recommendation quality using LLM-based or keyword-based ground truth
- **Metrics**: Precision, recall, F1 with detailed logging per user and cycle

### **PromptOptimizerAgent**

- **Primary Model**: Claude 3.5 Sonnet (strong creativity/analysis, 40% cheaper than GPT-4o)
- **Fallback**: Gemini 2.5 Pro, GPT-4o, Gemini Flash Thinking
- **Function**: Analyzes evaluation history and dynamically updates system prompts
- **Strategies**: Exploit (incremental improvement) and explore (diversification)

The **Orchestrator** coordinates these agents, running multi-cycle experiments, logging results, and managing prompt evolution.

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
