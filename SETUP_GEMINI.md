# Gemini Flash 优先模式设置指南

## 🎯 目标

确保在所有 API 都可用的情况下，RecommendationAgent 优先使用 Gemini 2.0 Flash。

## 📋 步骤

### 1. 创建环境配置文件

复制模板文件并创建你的 `.env` 文件：

```bash
cp env.template .env
```

### 2. 配置 API 密钥

编辑 `.env` 文件，填入你的 API 密钥：

```bash
# OpenAI API 配置
OPENAI_API_KEY=sk-your-openai-key-here

# Google Gemini API 配置
GOOGLE_API_KEY=your-google-api-key-here

# Anthropic Claude API 配置
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# 其他配置保持不变...
```

### 3. 获取 API 密钥

#### Google Gemini API Key

1. 访问 [Google AI Studio](https://makersuite.google.com/app/apikey)
2. 创建新的 API 密钥
3. 复制密钥到 `GOOGLE_API_KEY`

#### OpenAI API Key

1. 访问 [OpenAI Platform](https://platform.openai.com/api-keys)
2. 创建新的 API 密钥
3. 复制密钥到 `OPENAI_API_KEY`

#### Anthropic API Key

1. 访问 [Anthropic Console](https://console.anthropic.com/)
2. 创建新的 API 密钥
3. 复制密钥到 `ANTHROPIC_API_KEY`

### 4. 测试配置

运行测试脚本验证配置：

```bash
python test_api_keys.py
```

你应该看到：

```
✅ Google Gemini: 已配置
✅ OpenAI: 已配置
✅ Anthropic Claude: 已配置
```

### 5. 运行 Gemini 优先实验

使用专门的脚本运行实验：

```bash
python run_with_gemini.py
```

或者使用标准实验脚本：

```bash
python run_experiment.py --cycles 3 --users 15 --mode llm
```

## 🔍 验证 Gemini 使用

### 检查日志输出

在实验运行过程中，你应该看到：

```
使用 Gemini 2.0 Flash 作为推荐模型
```

### 检查结果文件

在 `logs/gemini_priority_*/summary.json` 中，确认：

```json
{
  "model_used": "gemini-2.0-flash-exp",
  "method_used": "llm_rerank"
}
```

## 📊 预期结果

### 模型选择优先级

1. **RecommendationAgent**: Gemini 2.0 Flash (优先)
2. **EvaluationAgent**: GPT-4o (优先)
3. **PromptOptimizerAgent**: GPT-4o (优先)

### 性能特点

- **速度**: Gemini Flash 响应更快
- **成本**: Gemini Flash 价格更低
- **质量**: 保持推荐质量

## 🚨 故障排除

### 问题 1: 仍然使用 GPT-4o

**原因**: GOOGLE_API_KEY 未正确设置
**解决**: 检查 `.env` 文件中的密钥配置

### 问题 2: Gemini API 调用失败

**原因**: API 密钥无效或配额不足
**解决**: 检查 Google AI Studio 中的配额和密钥状态

### 问题 3: 导入错误

**原因**: 缺少依赖包
**解决**: 运行 `pip install -r requirements.txt`

## 💡 提示

1. **成本控制**: Gemini Flash 比 GPT-4o 便宜很多
2. **速度优势**: Flash 模型响应速度更快
3. **质量保证**: 在推荐任务上表现良好
4. **符合要求**: 完全符合 Sekai 挑战的技术要求

## 📈 对比结果

运行完成后，你可以对比：

- `logs/precision_boost_3cycles_15user/` (GPT-4o 结果)
- `logs/gemini_priority_*/` (Gemini Flash 结果)

比较两者的性能差异和成本差异。
