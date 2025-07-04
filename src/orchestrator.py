# src/orchestrator.py
"""
Orchestrator
============
• 集成 Prompt‑Optimizer、RecommendationAgent、EvaluationAgent
• 命令行一键运行 ≥ N 轮循环，自动落盘日志
• 支持无 API‑Key 离线演示（自动使用 stub 模型）
• 记录提示词演化历史和详细评估结果

Usage
-----
# 3 轮、最小增益 0.01（默认值）
python -m src.orchestrator --cycles 3

# 全参数
python -m src.orchestrator \
  --cycles 5 \
  --min-delta 0.005 \
  --sample-users 5 \
  --log-dir logs/run_$(date +%s)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
from loguru import logger
import numpy as np

# ------------------ 将项目根目录放入 PYTHONPATH ------------------ #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------ 导入三大 Agent ------------------------ #
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.evaluation_agent import EvaluationAgent
from src.agents.prompt_optimizer import (
    PromptOptimizerAgent,
    OptimizeInput,
)

# ---------------------------- 常量 ------------------------------ #
DATA_DIR = PROJECT_ROOT / "data" / "raw"  # users.csv, contents.csv, interactions.csv

# 可配置的日志字段和评测指标
LOG_FIELDS = ["user_id", "precision_at_k", "recall_at_k", "f1_at_k", "recommended", "ground_truth"]
EVAL_METRICS = ["precision", "recall", "f1"]

# ------------------------ 数据结构定义 -------------------------- #
@dataclass
class CycleResult:
    """单轮循环的结果"""
    cycle: int
    precision_at_k: float
    recall_at_k: float
    prompt_version: str
    expected_gain: float
    actual_gain: float
    optimization_strategy: str
    timestamp: str
    sample_size: int
    user_results: List[Dict[str, Any]]


class PromptHistoryTracker:
    """跟踪和记录提示词演化历史"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.history = []
        self.prompt_file = log_dir / "prompt_evolution.json"
        
    def add_entry(self, cycle: int, prompt: str, metrics: Dict[str, Any], 
                  rationale: str = "", is_updated: bool = True, 
                  optimization_strategy: str = "none"):
        """记录一次提示词状态"""
        entry = {
            "cycle": cycle,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "metrics": metrics,
            "rationale": rationale,
            "is_updated": is_updated,
            "optimization_strategy": optimization_strategy,
            "version": f"v{cycle}.0"
        }
        self.history.append(entry)
        self._save()
        
    def _save(self):
        """保存到文件"""
        with open(self.prompt_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
            
    def get_best_prompt(self) -> Optional[Dict[str, Any]]:
        """获取最佳表现的提示词"""
        if not self.history:
            return None
        return max(self.history, key=lambda x: x['metrics'].get('precision_at_k', 0))
    
    def generate_report(self) -> str:
        """生成提示词演化报告"""
        report = ["# Prompt Evolution Report\n"]
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        for i, entry in enumerate(self.history):
            report.append(f"\n## Cycle {entry['cycle']} - {entry['timestamp']}")
            report.append(f"**Version**: {entry['version']}")
            report.append(f"**Strategy**: {entry['optimization_strategy']}")
            report.append(f"**Precision@10**: {entry['metrics'].get('precision_at_k', 0):.3f}")
            report.append(f"**Recall@10**: {entry['metrics'].get('recall_at_k', 0):.3f}")
            
            if i > 0:
                prev_score = self.history[i-1]['metrics'].get('precision_at_k', 0)
                improvement = entry['metrics'].get('precision_at_k', 0) - prev_score
                report.append(f"**Improvement**: {improvement:+.3f}")
            
            if entry['is_updated']:
                report.append(f"\n### Prompt:")
                report.append(f"```\n{entry['prompt']}\n```")
                
                if entry['rationale']:
                    report.append(f"\n### Optimization Rationale:")
                    report.append(entry['rationale'])
            else:
                report.append("\n*No prompt update in this cycle*")
                
        # 添加最佳提示词总结
        best = self.get_best_prompt()
        if best:
            report.append(f"\n## Best Performing Prompt")
            report.append(f"**Cycle**: {best['cycle']}")
            report.append(f"**Precision@10**: {best['metrics'].get('precision_at_k', 0):.3f}")
            report.append(f"**Strategy**: {best['optimization_strategy']}")
            report.append(f"```\n{best['prompt']}\n```")
            
        return "\n".join(report)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取三张原始 CSV，若缺失 interactions.csv 则返回空表"""
    users = pd.read_csv(DATA_DIR / "users.csv")
    contents = pd.read_csv(DATA_DIR / "contents.csv")
    interactions_path = DATA_DIR / "interactions.csv"
    if interactions_path.exists():
        interactions = pd.read_csv(interactions_path)
    else:
        interactions = pd.DataFrame(
            columns=["user_id", "content_id", "interaction_count"]
        )
    return users, contents, interactions


def should_stop(
    history: List[Dict[str, Any]],
    min_delta: float,
    patience: int = 2,
) -> bool:
    """若最近 patience+1 轮提升均 < min_delta，则停止"""
    if len(history) <= patience:
        return False
    recent = [h["precision_at_k"] for h in history[: patience + 1]]  # newest first
    deltas = [recent[i] - recent[i + 1] for i in range(patience)]
    return all(d < min_delta for d in deltas)


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算多个评估结果的平均指标"""
    if not results:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "std_precision": 0.0, "std_recall": 0.0}
    
    precisions = [r.get("precision_at_k", 0) for r in results]
    recalls = [r.get("recall_at_k", 0) for r in results]
    
    return {
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "std_precision": float(np.std(precisions, ddof=1)) if len(precisions) > 1 else 0.0,
        "std_recall": float(np.std(recalls, ddof=1)) if len(recalls) > 1 else 0.0,
    }


def main(
    cycles: int,
    min_delta: float,
    sample_users: int,
    log_dir: Path,
    use_llm_evaluation: bool = True,
    dry_run: bool = False,
    recall_mode: str = "llm",
    rerank_window: int = 30,
    eval_mode: str = "llm",
    use_simple_rerank: bool = False,
    tag_weight: float = 0.1,
    cold_start_boost: float = 0.2,
):
    print(f"[orchestrator.py] tag_weight={tag_weight}, cold_start_boost={cold_start_boost}, recall_mode={recall_mode}, rerank_window={rerank_window}")
    # 1. 载入数据
    users_df, contents_df, inter_df = load_data()
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "eval_history.jsonl").touch()

    # 2. 实例化 Agents
    logger.info("Initializing agents...")
    reco_agent = RecommendationAgent(
        contents_df, inter_df, dry_run=dry_run,
        recall_mode=recall_mode, rerank_window=rerank_window,
        use_simple_rerank=use_simple_rerank,
        tag_weight=tag_weight,
        cold_start_boost=cold_start_boost,
    )
    eval_agent = EvaluationAgent(
        users_df, contents_df, reco_agent,
        use_llm_for_ground_truth=use_llm_evaluation,
        use_llm_for_tag_simulation=use_llm_evaluation,
        eval_mode=eval_mode
    )
    try:
        from src.models.model_wrapper import create_optimizer_agent
        optimizer = PromptOptimizerAgent(model_wrapper=create_optimizer_agent())
    except Exception:
        from src.models.model_wrapper import create_recommendation_agent
        optimizer = PromptOptimizerAgent(model_wrapper=create_recommendation_agent())

    prompt_tracker = PromptHistoryTracker(log_dir)
    history = []
    detailed_results = []

    def _run_single_cycle(step: int, do_opt: bool):
        t0 = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Cycle {step} | {'baseline' if not do_opt else 'optimize'}")
        logger.info(f"{'='*60}")
        cycle_user_results = []
        logger.info(f"Evaluating with {sample_users} sampled users...")
        # --- 修复：每轮采样用户唯一 ---
        if sample_users > len(users_df):
            logger.warning(f"Requested sample_users={sample_users} > total users={len(users_df)}, using all users.")
            sampled_users = users_df.sample(frac=1, random_state=step).reset_index(drop=True)
        else:
            sampled_users = users_df.sample(n=sample_users, replace=False, random_state=step).reset_index(drop=True)
        used_user_ids = set()
        for _, urow in sampled_users.iterrows():
            user_id = urow.get("user_id")
            if user_id in used_user_ids:
                continue  # 冗余保护，理论上不会发生
            used_user_ids.add(user_id)
            user_dict = urow.to_dict()
            result = eval_agent.evaluate(user_dict)
            result_dict = asdict(result)
            result_dict["cycle"] = step
            result_dict["prompt_version"] = reco_agent.prompt_version
            cycle_user_results.append(result_dict)
        # prompt优化
        if do_opt and step >= 1:
            opt_out = optimizer.optimize(
                OptimizeInput(
                    eval_history=history[:10],
                    current_prompt=reco_agent.prompt_template,
                    min_delta=min_delta,
                )
            )
            is_updated = bool(opt_out.new_prompt)
            if is_updated:
                logger.info(f"✅ Prompt updated with strategy: {opt_out.optimization_strategy}")
                reco_agent.update_prompt_template(opt_out.new_prompt or "")
            else:
                logger.info("❌ No prompt update (expected gain too low or optimization failed)")
        else:
            opt_out = SimpleNamespace(
                new_prompt=None,
                expected_gain=0.0,
                optimization_strategy="baseline" if not do_opt else "initial",
                rationale="Baseline evaluation, no optimization." if not do_opt else "Initial cycle, no optimization."
            )
            is_updated = False
        # 将优化信息写入每个用户结果
        for result_dict in cycle_user_results:
            result_dict["optimization_strategy"] = opt_out.optimization_strategy
            result_dict["expected_gain"] = opt_out.expected_gain
        # 写入 eval_history.jsonl
        for result_dict in cycle_user_results:
            with open(log_dir / "eval_history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
        metrics = calculate_metrics(cycle_user_results)
        actual_gain = 0.0
        if history:
            actual_gain = metrics["precision_at_k"] - history[0]["precision_at_k"]
        cycle_result = CycleResult(
            cycle=step,
            precision_at_k=metrics["precision_at_k"],
            recall_at_k=metrics["recall_at_k"],
            prompt_version=reco_agent.prompt_version,
            expected_gain=opt_out.expected_gain,
            actual_gain=actual_gain,
            optimization_strategy=opt_out.optimization_strategy,
            timestamp=datetime.now().isoformat(),
            sample_size=sample_users,
            user_results=cycle_user_results
        )
        detailed_results.append(cycle_result)
        history.insert(0, {
            "cycle": step,
            "reports": cycle_user_results,
            "precision_at_k": metrics["precision_at_k"],
            "recall_at_k": metrics["recall_at_k"],
            "prompt_version": reco_agent.prompt_version,
            "expected_gain": opt_out.expected_gain,
            "optimization_strategy": opt_out.optimization_strategy,
        })
        prompt_tracker.add_entry(
            cycle=step,
            prompt=reco_agent.prompt_template,
            metrics=metrics,
            rationale=opt_out.rationale,
            is_updated=is_updated,
            optimization_strategy=opt_out.optimization_strategy
        )
        elapsed = time.time() - t0
        logger.info(
            "Cycle {:>2} | P@{} = {:.3f}±{:.3f} | R@{} = {:.3f}±{:.3f} | "
            "Δ = {:+.3f} (expected: {:+.3f}) | {} | {:.1f}s",
            step,
            eval_agent.k,
            metrics["precision_at_k"],
            metrics["std_precision"],
            eval_agent.k,
            metrics["recall_at_k"],
            metrics["std_recall"],
            actual_gain,
            opt_out.expected_gain,
            reco_agent.prompt_version,
            elapsed,
        )

    def finalize():
        logger.info(f"\n{'='*60}")
        logger.info("Finalizing results...")
        summary_path = log_dir / "summary.json"
        summary_data = [asdict(r) for r in detailed_results]
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        prompt_report = prompt_tracker.generate_report()
        report_path = log_dir / "prompt_evolution_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(prompt_report)
        generate_final_report(detailed_results, log_dir)
        logger.success(
            "Run finished ✓  |  Results saved to:\n"
            f"  - Summary: {summary_path}\n"
            f"  - Prompt evolution: {prompt_tracker.prompt_file}\n"
            f"  - Report: {report_path}\n"
            f"  - Detailed logs: {log_dir / 'eval_history.jsonl'}"
        )

    # —————————— 新增：始终先做一次评估 ——————————
    if cycles == 0:
        logger.info("cycles=0 ⇒ 仅做基线评估")
        _run_single_cycle(step=0, do_opt=False)
        finalize()
        return
    # ————————————————————————————————————————————

    # baseline eval
    _run_single_cycle(step=0, do_opt=False)
    # 后续 cycles 次正式优化循环
    for step in range(1, cycles + 1):
        _run_single_cycle(step, do_opt=True)
        # 早停判断
        if should_stop(history, min_delta):
            logger.info(f"Early-stop after cycle {step}")
            break
    finalize()


def generate_final_report(results: List[CycleResult], log_dir: Path):
    """生成最终的实验报告"""
    report = ["# Sekai Recommendation System - Experiment Summary\n"]
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if not results:
        report.append("No results to report.")
    else:
        # 总体性能
        initial_precision = results[0].precision_at_k
        final_precision = results[-1].precision_at_k
        improvement = final_precision - initial_precision
        
        report.append("## Overall Performance\n")
        report.append(f"- **Initial Precision@10**: {initial_precision:.3f}")
        report.append(f"- **Final Precision@10**: {final_precision:.3f}")
        if initial_precision > 0:
            report.append(f"- **Total Improvement**: {improvement:+.3f} ({improvement/initial_precision*100:+.1f}%)")
        else:
            report.append(f"- **Total Improvement**: {improvement:+.3f} (N/A - initial precision was 0)")
        report.append(f"- **Number of Cycles**: {len(results)}")
        report.append(f"- **Sample Size per Cycle**: {results[0].sample_size}")
        
        # 最佳性能
        best_cycle = max(results, key=lambda x: x.precision_at_k)
        report.append(f"\n## Best Performance\n")
        report.append(f"- **Cycle**: {best_cycle.cycle}")
        report.append(f"- **Precision@10**: {best_cycle.precision_at_k:.3f}")
        report.append(f"- **Recall@10**: {best_cycle.recall_at_k:.3f}")
        report.append(f"- **Optimization Strategy**: {best_cycle.optimization_strategy}")
        
        # 详细结果表
        report.append("\n## Detailed Results\n")
        report.append("| Cycle | Precision@10 | Recall@10 | Strategy | Expected Gain | Actual Gain |")
        report.append("|-------|-------------|-----------|----------|---------------|-------------|")
        
        for result in results:
            report.append(
                f"| {result.cycle} | {result.precision_at_k:.3f} | "
                f"{result.recall_at_k:.3f} | {result.optimization_strategy} | "
                f"{result.expected_gain:+.3f} | {result.actual_gain:+.3f} |"
            )
    
    # 保存报告
    report_path = log_dir / "experiment_summary.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))


# -------------------------- CLI 入口 ----------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sekai Recommendation System Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python -m src.orchestrator
  
  # Run 5 cycles with 10 users per cycle
  python -m src.orchestrator --cycles 5 --sample-users 10
  
  # Use keyword-based evaluation (faster, less accurate)
  python -m src.orchestrator --no-llm-eval
        """
    )
    
    parser.add_argument(
        "--cycles", 
        type=int, 
        default=3, 
        help="Maximum number of optimization cycles (default: 3)"
    )
    parser.add_argument(
        "--min-delta", 
        type=float, 
        default=0.01, 
        help="Minimum improvement threshold for early stopping (default: 0.01)"
    )
    parser.add_argument(
        "--sample-users", 
        type=int, 
        default=3, 
        help="Number of users to sample per evaluation cycle (default: 3)"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "logs" / f"run_{int(time.time())}",
        help="Directory for logs and results (default: logs/run_<timestamp>)"
    )
    parser.add_argument(
        "--no-llm-eval",
        action="store_true",
        help="Use keyword-based evaluation instead of LLM (faster but less accurate)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--recall-mode",
        type=str,
        default="llm",
        help="Recall mode for RecommendationAgent (default: llm)"
    )
    parser.add_argument(
        "--rerank-window",
        type=int,
        default=30,
        help="Rerank window for RecommendationAgent (default: 30)"
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="llm",
        help="Evaluation mode for EvaluationAgent (default: llm)"
    )
    parser.add_argument(
        "--use-simple-rerank",
        action="store_true",
        help="Use simple rerank for RecommendationAgent"
    )
    parser.add_argument(
        "--tag-weight",
        type=float,
        default=0.1,
        help="Tag weight for RecommendationAgent (default: 0.1)"
    )
    parser.add_argument(
        "--cold-start-boost",
        type=float,
        default=0.2,
        help="Cold start boost for RecommendationAgent (default: 0.2)"
    )

    args = parser.parse_args()
    
    # 配置日志
    logger.remove()  # 移除默认 handler
    logger.add(
        sys.stdout, 
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
    )
    
    # 同时记录到文件
    logger.add(
        args.log_dir / "orchestrator.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 运行主流程
    main(
        cycles=args.cycles,
        min_delta=args.min_delta,
        sample_users=args.sample_users,
        log_dir=args.log_dir,
        use_llm_evaluation=not args.no_llm_eval,
        recall_mode=args.recall_mode,
        rerank_window=args.rerank_window,
        eval_mode=args.eval_mode,
        use_simple_rerank=args.use_simple_rerank,
        tag_weight=args.tag_weight,
        cold_start_boost=args.cold_start_boost,
    )