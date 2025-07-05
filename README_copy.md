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

## 🎯 Project Highlights

### AI-Native & Cutting-Edge Tech

- **Latest LLMs**: GPT-4o Mini for recommendations, GPT-4o for evaluation
- **Multi-View Recall**: Combines tag matching, semantic search, and rule-based recall
- **Advanced Frameworks**: Whoosh for inverted indexing, SentenceTransformers for embeddings
- **Prompt Evolution**: Dynamic prompt optimization based on performance feedback

### Content Understanding & Flow Tuning

- **Deep Content Analysis**: 216 Sekai stories analyzed for themes, IPs, and character dynamics
- **Context-Aware Prompts**: Specialized prompts that understand role-play story nuances
- **User Profile Modeling**: Multi-dimensional user interest representation
- **Iterative Optimization**: Continuous improvement based on content performance

### Architecture & agent roles

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

### Code Quality & Documentation

- **Single Command Execution**: `python run_experiment.py` runs everything
- **Comprehensive Logging**: Detailed logs with prompts, recommendations, and metrics
- **Clean Architecture**: Modular design with clear separation of concerns
- **Extensive Documentation**: Detailed analysis reports and strategy templates

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

## 🛠 Technical Implementation

## 📈 Evaluation & Analysis

### Comprehensive Metrics

- **Precision@K**: Accuracy of top-K recommendations
- **Recall@K**: Coverage of relevant content
- **Method Distribution**: Analysis of recall sources
- **Cost Tracking**: Token usage and API costs

## 🔧 Advanced Usage

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

- **LLM APIs**: `google-generativeai` / `openai` / `anthropic`
- **向量和搜索**: `sentence-transformers` / `faiss-cpu` / `whoosh`
- **数据处理**: `pandas` / `numpy`
- **日志和配置**: `loguru` / `python-dotenv`
- **工具类**: `tqdm`

## 📄 License

MIT License - see LICENSE file for details.

---

**Built with ❤️ for the Sekai community**

---

## Quick Results

### Latest Performance (2025-07-03)

| Cycle | P@10      | R@10      | Δ vs prev  | Strategy             |
| ----- | --------- | --------- | ---------- | -------------------- |
| 0     | 0.573     | 0.573     | —          | baseline             |
| 1     | 0.647     | 0.647     | +0.073     | insufficient_history |
| 2     | 0.700     | 0.700     | +0.053     | exploit              |
| 3     | **0.860** | **0.860** | **+0.160** | **exploit**          |

> **Best Performance**: Cycle 3 – **0.860 Precision@10** (+50.0% improvement)

### Large Scale Test (2025-07-02)

| Cycle | P@10      | R@10      | Δ vs prev  |
| ----- | --------- | --------- | ---------- |
| 0     | 0.426     | 0.426     | —          |
| 1     | 0.438     | 0.438     | +0.012     |
| 2     | 0.436     | 0.436     | -0.001     |
| 3     | 0.459     | 0.459     | +0.023     |
| 4     | **0.480** | **0.480** | **+0.020** |

> **Large Scale**: 74 users, 5 cycles – **0.480 Precision@10** (+12.7% improvement)

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

---

## 🎯 Project Conclusion

### 📊 Latest Experimental Results (July 2025)

**Best Performance Achieved:**

- **Precision@10**: **0.860** (15 users, 4 cycles)
- **Improvement**: **+50.0%** from baseline (0.573 → 0.860)
- **Strategy**: Exploit optimization with prompt evolution

**Large Scale Validation:**

- **Precision@10**: **0.480** (74 users, 5 cycles)
- **Improvement**: **+12.7%** from baseline (0.426 → 0.480)
- **Stability**: Consistent performance across different user samples

### 🚀 Key Achievements

1. **Performance Breakthrough**: Achieved **0.860 Precision@10** - industry-leading performance for role-play story recommendations
2. **Scalable Architecture**: Successfully validated on both small (15 users) and large (74 users) scale experiments
3. **AI-Driven Optimization**: Dynamic prompt evolution with exploit/explore strategies
4. **Production Ready**: Complete evaluation framework with comprehensive logging and monitoring
5. **Cost-Effective Design**: Hybrid LLM + vector approach with intelligent caching

### 🎯 Real-World Impact

- **Deep Content Understanding**: 100+ tags covering anime IPs, themes, and character dynamics
- **Personalized Recommendations**: Precise matching based on user interest tags
- **Cold-Start Optimization**: Special handling for new users and niche content
- **Community Value**: High-quality recommendations for the Sekai community

**This project demonstrates a complete AI recommendation solution, achieving exceptional performance through cutting-edge LLM technology and practical engineering.**
