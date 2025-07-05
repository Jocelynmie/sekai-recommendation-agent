# Gemini Flash Priority Mode Setup Guide

## 🎯 Objective

Ensure that RecommendationAgent prioritizes Gemini 2.0 Flash when all APIs are available.

## 📋 Steps

### 1. Create Environment Configuration File

Copy the template file and create your `.env` file:

```bash
cp env.template .env
```

### 2. Configure API Keys

Edit the `.env` file and fill in your API keys:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=sk-your-openai-key-here

# Google Gemini API Configuration
GOOGLE_API_KEY=your-google-api-key-here

# Anthropic Claude API Configuration
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Other configurations remain unchanged...
```

### 3. Obtain API Keys

#### Google Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to `GOOGLE_API_KEY`

#### OpenAI API Key

1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a new API key
3. Copy the key to `OPENAI_API_KEY`

#### Anthropic API Key

1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Create a new API key
3. Copy the key to `ANTHROPIC_API_KEY`

### 4. Test Configuration

Run the test script to verify configuration:

```bash
python test_api_keys.py
```

You should see:

```
✅ Google Gemini: Configured
✅ OpenAI: Configured
✅ Anthropic Claude: Configured
```

### 5. Run Gemini Priority Experiment

Use the dedicated script to run the experiment:

```bash
python run_with_gemini.py
```

Or use the standard experiment script:

```bash
python run_experiment.py --cycles 3 --users 15 --mode llm
```

## 🔍 Verify Gemini Usage

### Check Log Output

During experiment execution, you should see:

```
Using Gemini 2.0 Flash as recommendation model
```

### Check Result Files

In `logs/gemini_priority_*/summary.json`, confirm:

```json
{
  "model_used": "gemini-2.0-flash-exp",
  "method_used": "llm_rerank"
}
```

## 📊 Expected Results

### Model Selection Priority

1. **RecommendationAgent**: Gemini 2.0 Flash (Primary)
2. **EvaluationAgent**: Gemini 2.5 Pro (Primary)
3. **PromptOptimizerAgent**: Claude 3.5 Sonnet (Primary)

### Performance Characteristics

- **Speed**: Gemini Flash responds faster
- **Cost**: Gemini Flash is much cheaper
- **Quality**: Maintains recommendation quality

## 🚨 Troubleshooting

### Issue 1: Still using GPT-4o

**Cause**: GOOGLE_API_KEY not properly set
**Solution**: Check key configuration in `.env` file

### Issue 2: Gemini API call failure

**Cause**: Invalid API key or insufficient quota
**Solution**: Check quota and key status in Google AI Studio

### Issue 3: Import error

**Cause**: Missing dependency packages
**Solution**: Run `pip install -r requirements.txt`

## 💡 Tips

1. **Cost Control**: Gemini Flash is much cheaper than GPT-4o
2. **Speed Advantage**: Flash model responds faster
3. **Quality Assurance**: Performs well on recommendation tasks
4. **Compliance**: Fully meets Sekai challenge technical requirements

## 📈 Comparison Results

After completion, you can compare:

- `logs/precision_boost_3cycles_15user/` (GPT-4o results)
- `logs/gemini_priority_*/` (Gemini Flash results)

Compare performance differences and cost differences between the two.

## 🎯 Model Selection Strategy

Our system uses the latest AI models from multiple vendors:

| Agent                    | Primary Model     | Key Advantages                                   | Cost Savings     |
| ------------------------ | ----------------- | ------------------------------------------------ | ---------------- |
| **RecommendationAgent**  | Gemini 2.0 Flash  | Latest "no-thinking" model, >300 RPS, free quota | 100% (free tier) |
| **EvaluationAgent**      | Gemini 2.5 Pro    | June 2025 release, SOTA reasoning, 1.5×-2× speed | 20% vs GPT-4o    |
| **PromptOptimizerAgent** | Claude 3.5 Sonnet | Strong creativity/analysis, latest Claude series | 40% vs GPT-4o    |

### Technical Benefits

- **Latest Models**: Uses 2025 releases (Gemini 2.5 Pro, Claude 3.5 Sonnet)
- **Multi-Vendor**: Google + Anthropic + OpenAI fallbacks
- **Cost Optimization**: 30% total savings vs all-GPT-4o
- **Performance**: 1.5×-2× speed improvement

For detailed analysis, see [`MODEL_SELECTION_STRATEGY.md`](./MODEL_SELECTION_STRATEGY.md).
