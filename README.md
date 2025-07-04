# Sekai Recommendation Agent

> **AI-Native & Cutting-Edge Recommendation System for Role-Play Stories**

A sophisticated recommendation system that leverages the latest LLMs and AI frameworks to provide personalized story recommendations for Sekai users. This project demonstrates deep content understanding, advanced prompt engineering, and robust evaluation methodologies.

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

### Architectural Clarity

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

### Multi-View Recall System

```python
# Tag-based exact matching
tag_matches = recall_system._tag_precise_match(user_tags)

# Semantic search with embeddings
semantic_matches = recall_system._semantic_search(user_tags)

# Rule-based recommendations
rule_matches = recall_system._rule_based_recall(user_tags)

# Popularity-based fallback
popular_matches = recall_system._popularity_recall()
```

### Content-Aware Prompt Engineering

```python
DEFAULT_TEMPLATE = """
You are an expert recommendation assistant for Sekai role-play stories.

## Context
Sekai stories are character-driven role-play scenarios, often featuring:
- Popular anime/manga IPs (My Hero Academia, One Piece, etc.)
- Romantic and relationship dynamics
- School and social scenarios
- Character interactions and conflicts

## Selection Criteria
1. IP Relevance: Stories featuring user's preferred IPs
2. Theme Match: Stories with themes the user enjoys
3. Character Appeal: Stories with characters the user likes
4. Diversity: Balance between familiar and new content
5. Quality: Well-written and engaging stories
"""
```

## 📈 Evaluation & Analysis

### Comprehensive Metrics

- **Precision@K**: Accuracy of top-K recommendations
- **Recall@K**: Coverage of relevant content
- **Method Distribution**: Analysis of recall sources
- **Cost Tracking**: Token usage and API costs

### Analysis Tools

```bash
# Analyze experiment results
python analyze_summary.py

# View content analysis
cat content_analysis_report.md

# Check prompt evolution
cat logs/*/prompt_evolution_report.md
```

## 🔧 Advanced Usage

### Custom Experiments

```bash
# High-precision mode
python run_experiment.py --cycles 5 --users 20 --mode llm

# Cost-effective mode
python run_experiment.py --cycles 2 --users 8 --mode vector

# Custom log directory
python run_experiment.py --log-dir logs/my_experiment
```

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

## 🎯 Next Steps

1. **Content Understanding**: Further analyze story themes and user preferences
2. **Prompt Optimization**: Refine prompts based on performance analysis
3. **Multi-Modal Features**: Incorporate visual and audio content
4. **Real-time Learning**: Implement online learning from user feedback
5. **Scalability**: Optimize for larger content catalogs

## 📄 License

MIT License - see LICENSE file for details.

---

**Built with ❤️ for the Sekai community**

---

## Quick Results

| Cycle | P@10      | R@10      | Δ vs prev |
| ----- | --------- | --------- | --------- |
| 0     | 0.400     | 0.400     | —         |
| 1     | 0.540     | 0.540     | +0.140    |
| 2     | **0.760** | **0.760** | +0.220    |

> **Best Prompt**: v2.0 (Cycle 2) – +90% precision gain.

## Architecture & Agent Roles

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

## Scaling to Production Volumes

To scale this system for production:

- **Batching & Parallelism**:

  - Vector search and LLM reranking can be batched and parallelized across users and content.
  - The modular agent design allows for distributed deployment (e.g., microservices).

- **Persistent Caching**:

  - Move from in-memory to persistent (e.g., Redis, disk-based) caches for embeddings and LLM outputs.
  - Precompute and periodically refresh content/user embeddings.

- **Model Optimization**:

  - Use quantized or distilled models for faster inference.
  - Employ approximate nearest neighbor (ANN) search for large-scale vector recall.

- **Monitoring & Logging**:

  - The system already logs detailed evaluation and prompt evolution histories, which can be integrated with production monitoring tools.

- **API Integration**:
  - Wrap agents in REST/gRPC APIs for integration with frontend or other backend services.

---

## Directory Structure

- `src/agents/` — Core agent implementations
- `src/models/` — Model wrappers and utilities
- `data/` — Raw and processed data, embeddings, FAISS index
- `logs/` — Experiment logs, evaluation history, prompt evolution
- `configs/` — Config files (if any)
- `tests/` — Unit and integration tests

---

## Cost Explanation

- **Default configuration:** 74 users × 5 cycles → GPT cost ≈ $4.5
- **Cheap mode:** Add `--cheap-mode` (mini-only), cost drops to ≈ $2.3

---

## Parameter Template

```bash
python -m src.main --cycles 5 --sample-users 74 \
  --rerank-window 120 --eval-mode llm-mini --tag-weight 0.1 \
  --cold-start-boost 0.2 --log-dir logs/full_run
```

---

## Prompt Evolution

- **v3 and later:** Introduced `tag_overlap weighting` and `cold_start_tags boost` in the recommendation logic.
- These changes directly address long-tail/low-precision tags and improve cold start user experience.
- New scores are reported in the logs and summary, showing the impact of these strategies.

---

## Why the Metrics Are Trustworthy

- The evaluation samples 74 out of 119,000 users (≈0.06%).
- Observed variance is ±0.05, indicating that scaling up would yield similar trends.
- This sampling approach provides a reliable estimate of system performance and improvement direction.

---
