# Model Selection Strategy Document

## 🎯 Overall Strategy

Based on the principles of **latest models first** and **optimal cost-effectiveness**, select the most suitable model for each Agent.

## 📊 Agent Model Configuration Table

| Agent                    | Selected Model        | Main Reasons                                                                                                | Estimated Cost/1k tokens |
| ------------------------ | --------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------ |
| **RecommendationAgent**  | **Gemini 2.0 Flash**  | • Latest "no-thinking" model (2025-05)<br>• Ultra-fast: >300 RPS<br>• Generous free quota<br>• Perfect for high-volume, simple tasks | Free tier available      |
| **EvaluationAgent**      | **Gemini 2.5 Pro**    | • Newest SOTA model (2025-06)<br>• Superior reasoning capabilities<br>• 128k context window<br>• 1.5-2× faster inference | $0.004 input / $0.012 output |
| **PromptOptimizerAgent** | **Claude 3.5 Sonnet** | • Exceptional at creative and analytical tasks<br>• Latest Claude series<br>• Strong prompt engineering capabilities<br>• Consistent performance | $0.003 input / $0.015 output |

## 🔍 Detailed Analysis

### 1. RecommendationAgent

**Choice**: Gemini 2.0 Flash
**Reasons**:

- **Speed Priority**: Meets Sekai challenge's "Must be fast" requirement
- **Latest Technology**: Most recent "no-thinking" model optimized for speed
- **Cost Efficiency**: Generous free quota makes it ideal for high-volume recommendations
- **Performance**: Excellent at straightforward ranking tasks

**Fallback Options**:
1. GPT-4o Mini (fast and affordable)
2. GPT-3.5-turbo (budget option)

### 2. EvaluationAgent

**Choice**: Gemini 2.5 Pro
**Reasons**:

- **Release Date**: June 2025, incorporating latest improvements
- **Reasoning Capability**: State-of-the-art performance for complex evaluation logic
- **Context Window**: 128k tokens handle full user profiles and story pools
- **Speed**: 1.5-2× faster than previous generation models

**Fallback Options**:
1. GPT-4o (strong reasoning)
2. Claude 3.5 Haiku (fast evaluation)
3. GPT-4 (reliable baseline)

### 3. PromptOptimizerAgent

**Choice**: Claude 3.5 Sonnet
**Reasons**:

- **Creative Excellence**: Claude models excel at generating diverse, creative prompts
- **Analysis Depth**: Strong at understanding patterns in evaluation history
- **Reliability**: Consistent performance in iterative optimization tasks
- **Latest Series**: Benefits from Anthropic's newest improvements

**Fallback Options**:
1. Gemini 2.5 Pro (strong analysis)
2. GPT-4o (creative generation)
3. Gemini Flash Thinking (quick iterations)

## 💰 Cost Analysis

### Single Round Experiment Cost Estimate (15 users, 3 rounds)

| Agent                | Model             | Estimated tokens/round | Cost/round | Total Cost |
| -------------------- | ----------------- | ---------------------- | ---------- | ---------- |
| RecommendationAgent  | Gemini 2.0 Flash  | 1000                   | $0.000*    | $0.000*    |
| EvaluationAgent      | Gemini 2.5 Pro    | 2000                   | $0.032     | $0.096     |
| PromptOptimizerAgent | Claude 3.5 Sonnet | 500                    | $0.009     | $0.027     |

**Total**: ~$0.123 per complete experiment  
*Note: Gemini Flash operates within free quota limits

## 🚀 Technical Advantages

### 1. Cutting-Edge Technology
- Leverages models released in 2025 (Gemini 2.5 Pro, Gemini 2.0 Flash)
- Incorporates latest Claude 3.5 series improvements
- Benefits from recent advances in inference speed and quality

### 2. Multi-Vendor Strategy
- **Google**: Gemini 2.5 Pro + 2.0 Flash
- **Anthropic**: Claude 3.5 Sonnet
- **OpenAI**: Available as fallback

### 3. Task-Optimized Selection
- Recommendation: Optimized for speed and volume
- Evaluation: Optimized for reasoning and accuracy
- Optimization: Optimized for creativity and analysis

## 🔧 Implementation Details

### Model Priority Logic

```python
# RecommendationAgent
1. Gemini 2.0 Flash (speed + free tier)
2. GPT-4o Mini (fast fallback)
3. GPT-3.5-turbo (budget fallback)

# EvaluationAgent
1. Gemini 2.5 Pro (latest reasoning)
2. GPT-4o (strong fallback)
3. Claude 3.5 Haiku (fast alternative)
4. GPT-4 (reliable baseline)

# PromptOptimizerAgent
1. Claude 3.5 Sonnet (creative excellence)
2. Gemini 2.5 Pro (analytical strength)
3. GPT-4o (versatile fallback)
4. Gemini Flash Thinking (rapid iteration)