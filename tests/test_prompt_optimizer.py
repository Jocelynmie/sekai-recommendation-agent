import sys
from pathlib import Path
from typing import List, Dict, Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.prompt_optimizer import (
    PromptOptimizerAgent,
    OptimizeInput,
)

# --------------------------- 离线固定模型 --------------------------- #
class DummyModel:
    model_name = "dummy-optimizer"

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        # 永远返回 "不更新" JSON
        return '{"new_prompt": null, "rationale": "looks fine", "expected_gain": 0.0}'

    def count_tokens(self, text: str) -> int:
        return len(text) // 4


# --------------------------- Unit Tests ----------------------------- #
@pytest.fixture
def optimizer():
    return PromptOptimizerAgent(model_wrapper=DummyModel())


def test_no_update(optimizer):
    hist: List[Dict[str, Any]] = [{"precision_at_k": 0.4}]
    out = optimizer.optimize(
        OptimizeInput(
            eval_history=hist,
            current_prompt="SELECT exactly {k} IDs…",
            min_delta=0.05,
        )
    )
    assert out.new_prompt is None
    assert out.expected_gain == 0.0
