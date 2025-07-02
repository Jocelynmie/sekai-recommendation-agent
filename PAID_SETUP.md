# Paid 版本设置指南

## 快速开始

### 1. 设置 API Key

```bash
# 方法1：环境变量
export OPENAI_API_KEY=your_openai_api_key_here

# 方法2：.env文件
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

### 2. 小流量验证（5 用户）

```bash
python src/main.py
```

### 3. 检查结果

```bash
ls logs/paid_run/
cat logs/paid_run/experiment_summary.md
```

## 配置说明

### 当前配置

- **推荐模型**: GPT-4o Mini (便宜快速)
- **评估模型**: GPT-4o (强大推理)
- **优化循环**: 已跳过 (cycles=0)
- **最佳 Prompt**: 已锁定 (P@10 ≈ 0.492)

### 费用估算

- **GPT-4o Mini**: $0.00015/1K input, $0.0006/1K output
- **GPT-4o**: $0.0025/1K input, $0.01/1K output
- **5 用户测试**: 预计 < $0.01

### 生产环境配置

#### 小流量验证 (当前)

```python
# src/main.py
cycles=0,  # 跳过优化
sample_users=5,  # 小流量
log_dir=Path("logs/paid_run")
```

#### 全量运行

```python
# 修改 src/main.py
cycles=0,  # 跳过优化
sample_users=40,  # 全量用户
log_dir=Path("logs/paid_run_full")
```

## 监控建议

### Token 用量监控 ✅

- 在 `model_wrapper.py` 中已启用 `log_tokens=True`
- 每轮打印 prompt+completion token
- 快速评估单次请求均价
- 实时显示成本统计

### 速率限制 ✅

- 设置 `MAX_REQUESTS_PER_MINUTE=60` 环境变量
- 捕获 RateLimitError → exponential back‑off
- 自动重试机制（默认 3 次）
- 智能等待时间计算

### 预算控制 ✅

- 在 `.env` 中设置 `OPENAI_BILLING_HARD_LIMIT=$50.00`
- 实时预算监控和警告
- 超限自动停止执行
- 详细成本统计报告

### 环境变量配置

```bash
# 速率限制
export MAX_REQUESTS_PER_MINUTE=60
export RATE_LIMIT_RETRY_ATTEMPTS=3

# 日志配置
export LOG_TOKENS=true
export LOG_COSTS=true

# 预算控制
export OPENAI_BILLING_HARD_LIMIT=50.00
```

## 故障排除

### 常见问题

1. **API Key 未设置**: 检查环境变量或.env 文件
2. **速率限制**: 降低并发或增加重试间隔
3. **Token 超限**: 检查 prompt 长度，必要时裁剪

### 调试命令

```bash
# 检查配置
python test_paid_setup.py

# 测试模型连接
python -c "from src.models.model_wrapper import create_recommendation_agent; print(create_recommendation_agent().model_name)"
```

## 性能对比

| 版本        | 模型                | P@10  | 成本 | 速度 |
| ----------- | ------------------- | ----- | ---- | ---- |
| Free (Stub) | Gemini Flash (模拟) | 0.444 | $0   | 快   |
| Paid        | GPT-4o Mini         | ~0.49 | 低   | 快   |
| Paid        | GPT-4o              | ~0.49 | 中   | 中   |

## 下一步

1. ✅ 设置 API Key
2. ✅ 运行小流量测试
3. ✅ 验证结果
4. 🔄 根据需要调整用户数量
5. 🔄 部署到生产环境
