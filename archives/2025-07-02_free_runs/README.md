# Free Run 日志归档 - 2025-07-02

## 归档内容

本目录包含 2025 年 7 月 2 日完成的所有免费版本评测日志，用于验证推荐系统性能稳定性。

## 目录结构

```
2025-07-02_free_runs/
├── free_run/              # 初始小样本评测 (15用户×4循环)
├── free_run_big/          # 大样本评测 (40用户×4循环)
├── free_run_seed1/        # 种子1评测 (40用户×3循环)
├── free_run_seed2/        # 种子2评测 (40用户×6循环)
├── free_run_seed3/        # 种子3评测 (40用户×6循环)
├── free_run_seed42/       # 种子42评测
├── free_run_seed78/       # 种子78评测
├── free_run_seed123/      # 种子123评测
└── free_run_seed245/      # 种子245评测
```

## 关键结果

### 性能稳定性验证

- **最终 P@10 均值**: 0.444
- **最终 P@10 标准差**: 0.009
- **最终 P@10 方差**: 0.000 (远小于 0.02 阈值)
- **最佳 P@10 均值**: 0.492
- **样本数**: 4 个种子

### 最佳 Prompt 锁定

- **版本**: v1.0 (optimized)
- **性能**: P@10 ≈ 0.492
- **位置**: `src/agents/recommendation_agent.py`
- **Git 标签**: v1.0-prod

## 文件说明

每个评测目录包含：

- `experiment_summary.md` - 实验摘要
- `summary.json` - 详细结果数据
- `eval_history.jsonl` - 评估历史
- `prompt_evolution.json` - Prompt 演化记录
- `prompt_evolution_report.md` - Prompt 演化报告

## 归档原因

1. **空间整理**: 清理 logs 目录，为生产环境做准备
2. **版本管理**: 标记重要的开发里程碑
3. **性能验证**: 保存稳定性验证的完整记录
4. **生产准备**: 为付费版本部署做准备

## 相关提交

- **Commit**: `2a6223d` - "chore: lock optimised prompt v1.0 and prepare for paid rollout"
- **Tag**: `v1.0-prod` - "First production prompt (P@10≈0.49)"

## 下一步

- 切换到付费版本进行生产部署
- 使用锁定的最佳 Prompt (v1.0)
- 启用费用与速率监控
- 开始正式的用户推荐服务
