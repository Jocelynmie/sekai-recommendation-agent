"""
Enhanced Prompt Optimizer Agent
Improved analysis capabilities and prompt generation strategies
"""

from __future__ import annotations

import json
import logging
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import numpy as np

from .base import BaseAgent
from src.models.model_wrapper import BaseModelWrapper

logger = logging.getLogger(__name__)


@dataclass
class OptimizeInput:
    """Input data for the optimizer"""
    eval_history: List[Dict[str, Any]]
    current_prompt: str
    min_delta: float = 0.01


@dataclass
class OptimizeOutput:
    """Output data from the optimizer"""
    new_prompt: Optional[str]
    expected_gain: float
    optimization_strategy: str
    rationale: str


# Optimizer strategy registry and registration decorator
OPTIMIZER_STRATEGY_REGISTRY = {}

def register_optimizer_strategy(name):
    def decorator(fn):
        OPTIMIZER_STRATEGY_REGISTRY[name] = fn
        return fn
    return decorator

@register_optimizer_strategy("default")
def default_optimize_strategy(history, **kwargs):
    # Directly call the original analyze_history logic
    # This is just a placeholder, actual call will pass self.analyze_history
    return None  # Handled internally by PromptOptimizerAgent


class PromptOptimizerAgent(BaseAgent):
    """
    In multi-round evaluation-optimization cycles, dynamically adjusts system prompts based on historical performance.
    Typical workflow:
        1. `optimize()` receives evaluation results history from several rounds
        2. Parse history -> call `analyze_history()` to get statistical analysis
        3. Based on analysis results and preset strategies, construct new system prompt
        4. Return to orchestrator, which injects into RecommendationAgent
    """

    DEBUG_FORCE_UPDATE = False  # Can be set to True during development, False for formal evaluation
    MIN_HISTORY_FOR_OPT = 2  # Skip optimization if less than this value

    def __init__(self, model_wrapper, name="PromptOptimizerAgent", config=None):
        super().__init__(name, model_wrapper, config)
        self.model = model_wrapper
        self.version = 1  # Integer type for easy incrementing
        self.last_strategy = None
        self.explore_interval = 3  # Explore every 3 rounds

    # --------------------------------------------------------------------- #
    # External Interface
    # --------------------------------------------------------------------- #
    def optimize(self, input_data: OptimizeInput, strategy="default") -> OptimizeOutput:
        """
        Attempt to rewrite prompt based on evaluation history.
        Supports hot-swappable strategies.
        """
        history = input_data.eval_history
        min_delta = input_data.min_delta
        if len(history) < self.MIN_HISTORY_FOR_OPT:
            return OptimizeOutput(
                new_prompt=None,
                expected_gain=0.0,
                optimization_strategy="insufficient_history",
                rationale=f"Need ≥{self.MIN_HISTORY_FOR_OPT} cycles, got {len(history)}."
            )
        analysis = self.analyze_history(history)
        expected_gain = analysis["expected_gain"]
        cycle = history[0]["cycle"] if "cycle" in history[0] else len(history)
        # New explore trigger logic: trigger after 2 consecutive rounds with no gain
        no_gain = False
        if len(history) >= 3:
            recent = [h["precision_at_k"] for h in history[:3]]
            deltas = [recent[i] - recent[i+1] for i in range(2)]
            if all(d < min_delta for d in deltas):
                no_gain = True
        if no_gain:
            strategy = "explore"
            new_prompt = self.build_prompt(analysis, strategy=strategy, top_p=0.9, shuffle_top_n=50)
            self.version += 1
            self.last_strategy = strategy
            rationale = f"Explore round: triggered after 2 rounds with < min_delta gain (cycle={cycle})"
            return OptimizeOutput(
                new_prompt=new_prompt,
                expected_gain=expected_gain,
                optimization_strategy=strategy,
                rationale=rationale
            )
        # Exploit round: generate new prompt only if expected_gain is sufficient
        if expected_gain >= min_delta:
            strategy = "exploit"
            new_prompt = self.build_prompt(analysis, strategy=strategy)
            self.version += 1
            self.last_strategy = strategy
            rationale = f"Exploit: expected_gain={expected_gain:.3f} >= min_delta={min_delta:.3f}"
            return OptimizeOutput(
                new_prompt=new_prompt,
                expected_gain=expected_gain,
                optimization_strategy=strategy,
                rationale=rationale
            )
        return OptimizeOutput(
            new_prompt=None,
            expected_gain=expected_gain,
            optimization_strategy="no_update",
            rationale=f"expected_gain={expected_gain:.3f} < min_delta={min_delta:.3f}"
        )

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        BaseAgent standard interface implementation
        """
        # Extract OptimizeInput parameters from input_data
        eval_history = input_data.get("eval_history", [])
        current_prompt = input_data.get("current_prompt", "")
        min_delta = input_data.get("min_delta", 0.01)
        
        optimize_input = OptimizeInput(
            eval_history=eval_history,
            current_prompt=current_prompt,
            min_delta=min_delta
        )
        
        result = self.optimize(optimize_input)
        
        # Convert to dictionary format
        return {
            "new_prompt": result.new_prompt,
            "expected_gain": result.expected_gain,
            "optimization_strategy": result.optimization_strategy,
            "rationale": result.rationale
        }

    # --------------------------------------------------------------------- #
    # Internal Implementation
    # --------------------------------------------------------------------- #
    def analyze_history(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert multi-round evaluation results into actionable statistical information
        Return structure example:
        {
            "mean_p": 0.54,
            "mean_r": 0.33,
            "std_p": 0.10,
            "std_r": 0.08,
            "expected_gain": 0.07,
            "failures": {
                "failed_tags": ["harem", "toxic", "yandere"],
                "hard_users": [2435915, 2370305]
            }
        }
        """
        WINDOW = 30  # User count window
        # History is appended in evaluation order, each entry is a single user's evaluation result
        if history is None:
            recent = []
        else:
            recent = list(history[:WINDOW])
        # Calculate average metrics
        if not recent:
            return {'precision_at_k': 0.0, 'recall_at_k': 0.0, 'std_precision': 0.0, 'std_recall': 0.0, 'expected_gain': 0.0}
        def safe_float(val):
            try:
                if val is None:
                    return 0.0
                return float(val)
            except Exception:
                return 0.0
        precisions = [safe_float(r.get('precision_at_k', 0.0)) for r in recent]
        recalls = [safe_float(r.get('recall_at_k', 0.0)) for r in recent]
        if not precisions:
            precisions = [0.0]
        if not recalls:
            recalls = [0.0]
        mean_p = statistics.mean(precisions)
        mean_r = statistics.mean(recalls)
        std_p = statistics.stdev(precisions) if len(precisions) > 1 else 0.0
        std_r = statistics.stdev(recalls) if len(recalls) > 1 else 0.0
        # Simplified: don't calculate deltas
        dp = 0.0  # Simplified: don't calculate delta
        dr = 0.0  # Simplified: don't calculate delta
        worst_tag = None  # Simplified: don't analyze worst tag
        failed_tags = []  # Simplified: don't analyze failed tags
        # Calculate expected gain (simplified heuristic)
        expected_gain = (mean_p + mean_r) / 2 * 0.1  # Assume 10% improvement potential
        return {
            'mean_p': mean_p,
            'mean_r': mean_r,
            'std_p': std_p,
            'std_r': std_r,
            'dp': dp,
            'dr': dr,
            'worst_tag': worst_tag,
            'failed_tags': failed_tags,
            'expected_gain': expected_gain
        }

    def build_prompt(self, analysis: dict, strategy: str = "exploit", top_p: float = None, shuffle_top_n: int = None) -> str:
        import random
        mean_p = analysis["mean_p"]
        mean_r = analysis["mean_r"]
        dp = analysis["dp"]
        dr = analysis["dr"]
        worst_tag = analysis["worst_tag"]
        failed_tags = analysis.get("failed_tags", [])
        version = f"v{self.version}"
        header = f"# {version}\n# Strategy: {strategy}\n"
        perf_summary = (
            f"### Performance Summary\n"
            f"- Current mean precision ≈ {mean_p:.3f}\n"
            f"- Current mean recall ≈ {mean_r:.3f}\n"
            f"- Δ precision vs prev ≈ {dp:.3f}\n"
            f"- Δ recall vs prev ≈ {dr:.3f}\n"
        )
        focus_instr = ""
        focus_tags: list = []
        if failed_tags:
            focus_tags = random.sample(failed_tags, min(3, len(failed_tags)))
            focus_instr = f"\n### Next Round Focus\n- Upweight stories with tags: {', '.join([repr(t) for t in focus_tags])} (recently underperformed)\n"
        elif worst_tag:
            focus_tags = [worst_tag]
            focus_instr = f"\n### Next Round Focus\n- Upweight stories with tag: '{worst_tag}' (recently underperformed)\n"
        coldstart_instr = ""
        if failed_tags:
            coldstart_instr = ("\n### Cold Start/Long-tail Issues\n"
                               f"- The following tags are associated with users who had low precision (<0.2): {', '.join(failed_tags)}.\n"
                               "- Please consider upweighting or diversifying recommendations for these tags to improve cold start/long-tail user experience.\n")
        explore_note = ""
        if strategy == "explore":
            explore_note = "\n### Explore Mode: Randomize candidate order, use high temperature, encourage diversity and surprise.\n"
            prompt = (
                header
                + "Given a list of user interest tags and candidate story summaries, "
                + "select **exactly {k}** story IDs that best match the user's interest. "
                + "Respond strictly as a JSON list of integers.\n\n"
                + perf_summary + focus_instr + coldstart_instr + explore_note
                + (f"\n# Explore: top_p={top_p}, shuffle_top_n={shuffle_top_n}" if top_p is not None and shuffle_top_n is not None else "")
            )
        else:
            prompt = (
                header
                + "You are a recommendation engine for role‑play stories. "
                + "Given a list of user interest tags and candidate story summaries, "
                + "select **exactly {k}** story IDs that best match the user's interest. "
                + "Respond strictly as a JSON list of integers.\n\n"
                + perf_summary + focus_instr + coldstart_instr
            )
        # New: record focus_tags to self.last_focus_tags for logging/summary use
        self.last_focus_tags = focus_tags
        return prompt

    # --------------------------------------------------------------------- #
    # Debug / Visualization Helpers
    # --------------------------------------------------------------------- #
    def _debug_dump(self, analysis: Dict[str, Any], outfile: Path | None = None):
        """
        Output analysis results as JSON for offline debugging.
        """
        if outfile is None:
            outfile = Path("prompt_opt_analysis.json")
        outfile.write_text(json.dumps(analysis, indent=2, ensure_ascii=False))
        logger.info("Analysis dumped to %s", outfile.resolve())


# --------------------------- Simple Self-Test --------------------------- #
if __name__ == "__main__":
    dummy_history = [
        {
            "cycle": 0,
            "reports": [
                {
                    "user_id": 111,
                    "precision": 0.3,
                    "recall": 0.2,
                    "tags": ["yandere", "harem"],
                },
                {
                    "user_id": 222,
                    "precision": 0.6,
                    "recall": 0.5,
                    "tags": ["romance"],
                },
            ],
        },
        {
            "cycle": 1,
            "reports": [
                {
                    "user_id": 111,
                    "precision": 0.4,
                    "recall": 0.3,
                    "tags": ["harem"],
                },
                {
                    "user_id": 333,
                    "precision": 0.2,
                    "recall": 0.1,
                    "tags": ["toxic", "yandere"],
                },
            ],
        },
    ]

    class DummyModel(BaseModelWrapper):
        def __init__(self):
            super().__init__("dummy-model", 0.7)
            self.system_prompt = "Initial prompt"

        def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
            return f"Dummy response to: {prompt}"

        def count_tokens(self, text: str) -> int:
            return len(text) // 4

        def update_system_prompt(self, new_prompt: str):
            self.system_prompt = new_prompt

    agent = PromptOptimizerAgent(model_wrapper=DummyModel(), name="opt")
    agent.optimize(OptimizeInput(eval_history=dummy_history, current_prompt="test", min_delta=0.0))
    print("=== New System Prompt ===")
    print(agent.get_system_prompt())
    print("=========================")

    # Output analysis JSON
    result = agent.analyze_history(dummy_history)
    agent._debug_dump(result)
    print(f"ΔP,R ≈ {result['expected_gain']:.3f}")
    print(f"Strategy: {result.get('optimization_strategy')}")
