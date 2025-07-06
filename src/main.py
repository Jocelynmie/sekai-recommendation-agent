"""
Main Entry Point
Entry point for the recommendation system
"""

import argparse
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Ensure environment variables are loaded before importing other modules
load_dotenv(dotenv_path=".env", override=True)

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.orchestrator import main as orchestrator_main

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Sekai Recommendation System")
    parser.add_argument("--cycles", type=int, default=1, help="Number of optimization cycles")
    parser.add_argument("--sample-users", type=int, default=5, help="Number of users to sample")
    parser.add_argument("--min-delta", type=float, default=0.01, help="Minimum improvement threshold")
    parser.add_argument("--log-dir", type=str, default="logs/paid_run", help="Log directory")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true", help="Use recall only, skip LLM (for fast self-check)")
    parser.add_argument("--recall-mode", type=str, choices=["vector", "llm"], default="llm", help="Recall strategy: vector (pure vector) or llm (vector+LLM rerank)")
    parser.add_argument("--rerank-window", type=int, default=30, help="Number of candidates to send to LLM for rerank (default: 30)")
    parser.add_argument("--eval-mode", type=str, choices=["llm", "keyword"], default="llm", help="Evaluation mode for ground-truth: llm or keyword")
    parser.add_argument("--use-simple-rerank", action="store_true", help="Use simple rerank prompt for LLM rerank")
    parser.add_argument("--tag-weight", type=float, default=0.1, help="Weight for tag overlap boosting")
    parser.add_argument("--cold-start-boost", type=float, default=0.2, help="Boost for cold start tags")
    
    args = parser.parse_args()
    print(f"[main.py] args: {args}")
    
    # Debug environment variables
    logger.info("Environment check:")
    logger.info(f"  OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
    logger.info(f"  GOOGLE_API_KEY: {'Set' if os.getenv('GOOGLE_API_KEY') else 'Not set'}")
    logger.info(f"  ANTHROPIC_API_KEY: {'Set' if os.getenv('ANTHROPIC_API_KEY') else 'Not set'}")
    
    # Set random seed
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    orchestrator_main(
        cycles=args.cycles,
        min_delta=args.min_delta,
        sample_users=args.sample_users,
        log_dir=Path(args.log_dir),
        use_llm_evaluation=(args.eval_mode == "llm"),
        dry_run=args.dry_run,
        recall_mode=args.recall_mode,
        rerank_window=args.rerank_window,
        eval_mode=args.eval_mode,
        use_simple_rerank=args.use_simple_rerank,
        tag_weight=args.tag_weight,
        cold_start_boost=args.cold_start_boost,
    )

if __name__ == '__main__':
    main()
