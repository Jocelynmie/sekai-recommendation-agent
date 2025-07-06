#!/usr/bin/env python3
"""
Run Recommendation System Experiment
Main entry point for running experiments
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

from loguru import logger


def run_experiment(
    cycles: int = 3,
    users: int = 15,
    mode: str = "llm",
    min_delta: float = 0.01,
    log_dir: str = None,
    dry_run: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Run complete recommendation system experiment"""
    
    # Build command
    cmd = [
        sys.executable, "-m", "src.orchestrator",
        "--cycles", str(cycles),
        "--users", str(users),
        "--min-delta", str(min_delta),
    ]
    
    if mode == "vector":
        cmd.extend(["--recall-mode", "vector", "--eval-mode", "keyword"])
    elif mode == "llm":
        cmd.extend(["--recall-mode", "llm", "--eval-mode", "llm"])
    
    if log_dir:
        cmd.extend(["--log-dir", log_dir])
    
    if dry_run:
        cmd.append("--dry-run")
    
    # Add additional kwargs
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run experiment
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Experiment completed successfully")
        return {"success": True, "output": result.stdout}
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment failed: {e}")
        print(f"Error output: {e.stderr}")
        return {"success": False, "error": e.stderr}


def analyze_results(log_dir: str) -> Dict[str, Any]:
    """Analyze experiment results"""
    log_path = Path(log_dir)
    
    if not log_path.exists():
        return {"error": f"Log directory {log_dir} does not exist"}
    
    # Read summary.json
    summary_file = log_path / "summary.json"
    if not summary_file.exists():
        return {"error": "summary.json not found"}
    
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        
        # Calculate statistics
        if not summary_data:
            return {"error": "No data in summary.json"}
        
        precisions = [cycle.get("precision_at_k", 0) for cycle in summary_data]
        recalls = [cycle.get("recall_at_k", 0) for cycle in summary_data]
        
        analysis = {
            "total_cycles": len(summary_data),
            "final_precision": precisions[-1] if precisions else 0,
            "final_recall": recalls[-1] if recalls else 0,
            "best_precision": max(precisions) if precisions else 0,
            "best_recall": max(recalls) if recalls else 0,
            "improvement": precisions[-1] - precisions[0] if len(precisions) > 1 else 0,
            "cycles_data": summary_data
        }
        
        return analysis
        
    except Exception as e:
        return {"error": f"Failed to analyze results: {e}"}


def print_results(analysis: Dict[str, Any]):
    """Print experiment results"""
    if "error" in analysis:
        print(f"❌ Analysis failed: {analysis['error']}")
        return
    
    print("\n" + "="*60)
    print("📊 EXPERIMENT RESULTS")
    print("="*60)
    
    print(f"Total Cycles: {analysis['total_cycles']}")
    print(f"Final Precision@10: {analysis['final_precision']:.3f}")
    print(f"Final Recall@10: {analysis['final_recall']:.3f}")
    print(f"Best Precision@10: {analysis['best_precision']:.3f}")
    print(f"Best Recall@10: {analysis['best_recall']:.3f}")
    print(f"Total Improvement: {analysis['improvement']:+.3f}")
    
    # Show per-round results
    print("\n📈 Per-Round Results:")
    print("Cycle | Precision@10 | Recall@10 | Strategy")
    print("------|--------------|-----------|----------")
    
    for cycle_data in analysis['cycles_data']:
        cycle = cycle_data.get('cycle', '?')
        precision = cycle_data.get('precision_at_k', 0)
        recall = cycle_data.get('recall_at_k', 0)
        strategy = cycle_data.get('optimization_strategy', 'unknown')
        print(f"{cycle:5} | {precision:11.3f} | {recall:9.3f} | {strategy}")


def main():
    parser = argparse.ArgumentParser(
        description="Run recommendation system experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (1 cycle, 5 users)
  python run_experiment.py --cycles 1 --users 5
  
  # Full experiment (3 cycles, 15 users, LLM mode)
  python run_experiment.py --cycles 3 --users 15 --mode llm
  
  # Vector mode experiment (faster, lower cost)
  python run_experiment.py --cycles 2 --users 10 --mode vector
        """
    )
    
    parser.add_argument("--cycles", type=int, default=3, help="Number of optimization cycles")
    parser.add_argument("--users", type=int, default=15, help="Number of users to sample per cycle")
    parser.add_argument("--mode", choices=["llm", "vector"], default="llm", 
                       help="Experiment mode: llm (default) or vector")
    parser.add_argument("--min-delta", type=float, default=0.01, 
                       help="Minimum improvement threshold for early stopping")
    parser.add_argument("--log-dir", type=str, help="Custom log directory")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (no API calls)")
    parser.add_argument("--analyze-only", type=str, help="Only analyze existing results from log directory")
    
    args = parser.parse_args()
    
    # Check environment
    if not Path("data/processed").exists():
        print("❌ Error: data/processed directory not found")
        print("Please ensure you have run data preprocessing first")
        sys.exit(1)
    
    # Run experiment
    if args.analyze_only:
        analysis = analyze_results(args.analyze_only)
        print_results(analysis)
    else:
        result = run_experiment(
            cycles=args.cycles,
            users=args.users,
            mode=args.mode,
            min_delta=args.min_delta,
            log_dir=args.log_dir,
            dry_run=args.dry_run
        )
        
        if result["success"]:
            # Auto-analyze results
            if args.log_dir:
                analysis = analyze_results(args.log_dir)
                print_results(analysis)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main() 