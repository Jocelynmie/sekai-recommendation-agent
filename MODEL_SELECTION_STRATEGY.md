# Model Selection Strategy Document

## 🎯 Overall Strategy

Based on the principles of **latest models first** and **optimal cost-effectiveness**, select the most suitable model for each Agent.

## 📊 Agent Model Configuration Table

| Agent                    | Current (Primary) | New Primary           | Main Reasons                                                                                                                  | Estimated Cost Diff\* |
| ------------------------ | ----------------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| **EvaluationAgent**      | GPT‑4o (128k)     | **Gemini 2.5 Pro**    | • 2025‑06 release, SOTA reasoning<br>• 1.5×‑2× speed<br>• 128k ctx<br>• Input $0.004/k, Output $0.012/k (20% cheaper than 4o) | ↓ ≈$0.05              |
| **PromptOptimizerAgent** | GPT‑4o            | **Claude 3.5 Sonnet** | Latest Claude series, strong creativity/analysis; token price 40% lower than 4o                                               | ↓ ≈$0.01              |
| **RecommendationAgent**  | Gemini 2.0 Flash  | **Unchanged**         | Flash remains the latest "no-thinking" model from 2025‑05; >300 RPS & generous free quota                                     | -                     |

## 🔍 Detailed Analysis

### 1. EvaluationAgent

**Choice**: Gemini 2.5 Pro
**Reasons**:

- **Release Date**: June 2025, latest SOTA model
- **Performance Advantage**: 1.5×-2× speed improvement
- **Context**: 128k tokens, sufficient for complex evaluation tasks
- **Cost Advantage**: 20% cheaper than GPT-4o
  - Input: $0.004/k tokens
  - Output: $0.012/k tokens

**Fallback Options**:

1. GPT-4o (if Gemini unavailable)
2. GPT-4 (standard version)
3. Claude 3.5 Haiku (fast and cheap)

### 2. PromptOptimizerAgent

**Choice**: Claude 3.5 Sonnet
**Reasons**:

- **Creative Capability**: Claude series excels in creative and analytical tasks
- **Cost Advantage**: Token price 40% lower than GPT-4o
- **Stability**: Anthropic models perform consistently in prompt optimization tasks
- **Latest Technology**: Uses the latest Claude 3.5 series

**Fallback Options**:

1. Gemini 2.5 Pro (latest Google model)
2. GPT-4o (latest OpenAI)
3. Gemini Flash Thinking (fast thinking)

### 3. RecommendationAgent

**Choice**: Gemini 2.0 Flash
**Reasons**:

- **Speed Priority**: Meets Sekai challenge's "Must be fast" requirement
- **Cost Efficiency**: Generous free quota, >300 RPS
- **Latest Technology**: Latest "no-thinking" model released in May 2025
- **Stability**: Performs well in recommendation tasks

**Fallback Options**:

1. GPT-4o Mini (fast and cheap)
2. GPT-3.5-turbo (cheapest)

## 💰 Cost Analysis

### Single Round Experiment Cost Estimate (15 users, 3 rounds)

| Agent                | Model             | Estimated tokens/round | Cost/round | Total Cost |
| -------------------- | ----------------- | ---------------------- | ---------- | ---------- |
| EvaluationAgent      | Gemini 2.5 Pro    | 2000                   | $0.032     | $0.096     |
| PromptOptimizerAgent | Claude 3.5 Sonnet | 500                    | $0.009     | $0.027     |
| RecommendationAgent  | Gemini 2.0 Flash  | 1000                   | $0.000\*   | $0.000\*   |

**Total**: ~$0.123 per complete experiment \*Note: Gemini Flash has free quota

### Cost Savings Comparison

Compared to all-GPT-4o configuration:

- **EvaluationAgent**: ~20% savings ($0.05/round)
- **PromptOptimizerAgent**: ~40% savings ($0.01/round)
- **Total Savings**: ~$0.18 per complete experiment

## 🚀 Technical Advantages

### 1. Cutting-Edge Technology

- Uses Gemini 2.5 Pro released in June 2025
- Adopts latest Claude 3.5 Sonnet
- Maintains Gemini 2.0 Flash speed advantage

### 2. Multi-Vendor Comparison

- **Google**: Gemini 2.5 Pro + 2.0 Flash
- **Anthropic**: Claude 3.5 Sonnet
- **OpenAI**: As fallback option

### 3. Performance Optimization

- Evaluation Agent: Latest reasoning capability
- Optimization Agent: Best creative analysis
- Recommendation Agent: Fastest response speed

## 🔧 Implementation Details

### Model Priority Logic

```python
# EvaluationAgent
1. Gemini 2.5 Pro (latest, 20% cheaper)
2. GPT-4o (fallback)
3. GPT-4 (standard)
4. Claude 3.5 Haiku (fast)

# PromptOptimizerAgent
1. Claude 3.5 Sonnet (strong creativity, 40% cheaper)
2. Gemini 2.5 Pro (latest)
3. GPT-4o (fallback)
4. Gemini Flash Thinking (fast)

# RecommendationAgent
1. Gemini 2.0 Flash (fastest, free)
2. GPT-4o Mini (fallback)
3. GPT-3.5-turbo (cheapest)
```
