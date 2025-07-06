# src/orchestrator.py
"""
Orchestrator for Recommendation System
Manages the complete experiment workflow
"""

import json
import logging
import random
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from types import SimpleNamespace

# 3 rounds, minimum gain 0.01 (default)
# Full parameters
from loguru import logger

# ------------------ Add project root to PYTHONPATH ------------------ #
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ------------------------ Import Three Major Agents ------------------------ #
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.evaluation_agent import EvaluationAgent
from src.agents.prompt_optimizer import PromptOptimizerAgent, OptimizeInput

# ---------------------------- Constants ------------------------------ #
# Configurable log fields and evaluation metrics
EVAL_FIELDS = ["precision_at_k", "recall_at_k", "method_used", "model_used"]

# ------------------------ Data Structure Definitions -------------------------- #
@dataclass
class CycleResult:
    """Single cycle result"""
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
    """Track and record prompt evolution history"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.history = []
        self.prompt_file = log_dir / "prompt_evolution.json"
        
    def add_entry(self, cycle: int, prompt: str, metrics: Dict[str, Any], 
                  rationale: str = "", is_updated: bool = True, 
                  optimization_strategy: str = "none"):
        """Record one prompt state"""
        entry = {
            "cycle": cycle,
            "timestamp": time.time(),
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
        """Save to file"""
        with open(self.prompt_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
            
    def get_best_prompt(self) -> Optional[Dict[str, Any]]:
        """Get best performing prompt"""
        if not self.history:
            return None
        return max(self.history, key=lambda x: x['metrics'].get('precision_at_k', 0))
    
    def generate_report(self) -> str:
        """Generate prompt evolution report"""
        report = ["# Prompt Evolution Report\n"]
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
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
                
        # Add best prompt summary
        best = self.get_best_prompt()
        if best:
            report.append(f"\n## Best Performing Prompt")
            report.append(f"**Cycle**: {best['cycle']}")
            report.append(f"**Precision@10**: {best['metrics'].get('precision_at_k', 0):.3f}")
            report.append(f"**Strategy**: {best['optimization_strategy']}")
            report.append(f"```\n{best['prompt']}\n```")
            
        return "\n".join(report)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read three original CSVs, return empty table if interactions.csv missing"""
    import pandas as pd
    
    data_dir = Path("data/processed")
    
    # Load users and contents
    users_df = pd.read_csv(data_dir / "users.csv")
    contents_df = pd.read_csv(data_dir / "contents.csv")
    
    # Load interactions if available
    interactions_path = data_dir / "interactions.csv"
    if interactions_path.exists():
        inter_df = pd.read_csv(interactions_path)
    else:
        inter_df = pd.DataFrame(columns=["user_id", "content_id", "rating", "timestamp"])
    
    return users_df, contents_df, inter_df


def should_stop(history: List[Dict[str, Any]], min_delta: float, patience: int = 3) -> bool:
    """Stop if recent patience+1 rounds all have improvement < min_delta"""
    if len(history) < patience + 1:
        return False
    
    recent_improvements = []
    for i in range(patience):
        if i + 1 < len(history):
            improvement = history[i]["precision_at_k"] - history[i + 1]["precision_at_k"]
            recent_improvements.append(improvement)
    
    return all(imp < min_delta for imp in recent_improvements)


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate average metrics from multiple evaluation results"""
    if not results:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "std_precision": 0.0, "std_recall": 0.0}
    
    precisions = [r.get("precision_at_k", 0.0) for r in results]
    recalls = [r.get("recall_at_k", 0.0) for r in results]
    
    return {
        "precision_at_k": statistics.mean(precisions),
        "recall_at_k": statistics.mean(recalls),
        "std_precision": statistics.stdev(precisions) if len(precisions) > 1 else 0.0,
        "std_recall": statistics.stdev(recalls) if len(recalls) > 1 else 0.0
    }


def generate_final_report(detailed_results: List[CycleResult], log_dir: Path):
    """Generate final experiment report"""
    report_lines = [
        "# Recommendation System Experiment Report\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"Total Cycles: {len(detailed_results)}\n",
        f"Sample Size: {detailed_results[0].sample_size if detailed_results else 0}\n\n"
    ]
    
    # Overall performance
    if detailed_results:
        best_cycle = max(detailed_results, key=lambda x: x.precision_at_k)
        report_lines.extend([
            "## Overall Performance\n",
            f"Best Precision@10: {best_cycle.precision_at_k:.3f} (Cycle {best_cycle.cycle})\n",
            f"Best Recall@10: {best_cycle.recall_at_k:.3f} (Cycle {best_cycle.cycle})\n\n"
        ])
    
    # Best performance
    if detailed_results:
        report_lines.extend([
            "## Best Performance\n",
            f"Cycle: {best_cycle.cycle}\n",
            f"Precision@10: {best_cycle.precision_at_k:.3f}\n",
            f"Recall@10: {best_cycle.recall_at_k:.3f}\n",
            f"Strategy: {best_cycle.optimization_strategy}\n\n"
        ])
    
    # Detailed results table
    report_lines.append("## Detailed Results\n")
    report_lines.append("| Cycle | Precision@10 | Recall@10 | Strategy | Expected Gain | Actual Gain |")
    report_lines.append("|-------|--------------|-----------|----------|---------------|-------------|")
    
    for result in detailed_results:
        report_lines.append(
            f"| {result.cycle} | {result.precision_at_k:.3f} | {result.recall_at_k:.3f} | "
            f"{result.optimization_strategy} | {result.expected_gain:.3f} | {result.actual_gain:.3f} |"
        )
    
    # Save report
    report_path = log_dir / "experiment_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))


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
    # 1. Load data
    users_df, contents_df, inter_df = load_data()
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "eval_history.jsonl").touch()

    # 2. Instantiate Agents
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
        logger.info(f"Starting cycle {step}...")
        
        # Fix: unique user sampling per round
        available_users = users_df['user_id'].tolist()
        if len(available_users) < sample_users:
            logger.warning(f"Only {len(available_users)} users available, using all")
            selected_users = available_users
        else:
            selected_users = random.sample(available_users, sample_users)
        
        cycle_user_results = []
        for user_id in selected_users:
            if user_id not in users_df['user_id'].values:
                continue  # Redundant protection, theoretically shouldn't happen
            
            result = eval_agent.evaluate_user(user_id)
            cycle_user_results.append(asdict(result))

        # Prompt optimization
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
        # Write optimization info to each user result
        for result_dict in cycle_user_results:
            result_dict["optimization_strategy"] = opt_out.optimization_strategy
            result_dict["expected_gain"] = opt_out.expected_gain
        # Write to eval_history.jsonl
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
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
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

    # —————————— New: always do baseline evaluation first ——————————
    if cycles == 0:
        logger.info("cycles=0 ⇒ baseline evaluation only")
        _run_single_cycle(step=0, do_opt=False)
        finalize()
        return
    # ———————————————————————————————————————————————————

    # Baseline eval
    _run_single_cycle(step=0, do_opt=False)
    # Subsequent cycles formal optimization loops
    for step in range(1, cycles + 1):
        _run_single_cycle(step, do_opt=True)
        # Early stop judgment
        if should_stop(history, min_delta):
            logger.info(f"Early-stop after cycle {step}")
            break
    finalize()


# -------------------------- CLI Entry ----------------------------- #
if __name__ == "__main__":
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser(description="Run recommendation system experiment")
    parser.add_argument("--cycles", type=int, default=3, help="Number of optimization cycles")
    parser.add_argument("--min-delta", type=float, default=0.01, help="Minimum improvement threshold")
    parser.add_argument("--users", type=int, default=10, help="Number of users to sample")
    parser.add_argument("--log-dir", type=str, default="logs/experiment", help="Log directory")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--recall-mode", choices=["llm", "vector"], default="llm", help="Recall mode")
    parser.add_argument("--rerank-window", type=int, default=30, help="Rerank window size")
    parser.add_argument("--eval-mode", choices=["llm", "keyword", "vector"], default="llm", help="Evaluation mode")
    parser.add_argument("--use-simple-rerank", action="store_true", help="Use simple rerank template")
    parser.add_argument("--tag-weight", type=float, default=0.1, help="Tag weight for fusion")
    parser.add_argument("--cold-start-boost", type=float, default=0.2, help="Cold start boost factor")
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    # Also log to file
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / "orchestrator.log", rotation="10 MB")
    
    # Set random seed
    random.seed(42)
    
    # Run main process
    main(
        cycles=args.cycles,
        min_delta=args.min_delta,
        sample_users=args.users,
        log_dir=log_dir,
        dry_run=args.dry_run,
        recall_mode=args.recall_mode,
        rerank_window=args.rerank_window,
        eval_mode=args.eval_mode,
        use_simple_rerank=args.use_simple_rerank,
        tag_weight=args.tag_weight,
        cold_start_boost=args.cold_start_boost,
    )