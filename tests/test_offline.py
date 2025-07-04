#!/usr/bin/env python
"""
离线测试脚本 - 不使用任何收费 API
仅用于验证代码逻辑是否正常工作
"""

import sys
import os
from pathlib import Path

# 设置环境变量，强制使用 stub 模型
os.environ["USE_STUB_MODELS"] = "true"
os.environ["GOOGLE_API_KEY"] = ""  # 清空 API keys
os.environ["OPENAI_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestrator import main
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.evaluation_agent import EvaluationAgent
from src.agents.prompt_optimizer import PromptOptimizerAgent
from src.models.gemini_wrapper import GeminiWrapper
import pandas as pd
from loguru import logger
import time
import json


class StubModelWrapper:
    """完全离线的模型桩"""
    
    def __init__(self, model_name="stub-model"):
        self.model_name = model_name
        self.call_count = 0
        
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        self.call_count += 1
        
        # 模拟不同的响应
        if "select 3-7 tags" in prompt.lower():
            return "romance, action, fantasy"
        elif "select exactly" in prompt.lower() and "story ids" in prompt.lower():
            return "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
        elif "improve the prompt template" in prompt.lower():
            # 模拟优化建议
            return json.dumps({
                "new_prompt": "You are a recommendation engine. Given user tags and stories, select exactly {k} story IDs that best match. Focus on: 1) Direct tag matches 2) Theme alignment 3) Character preferences. Return ONLY a JSON array of integers.",
                "rationale": "Simplified and clarified the instructions",
                "expected_gain": 0.05,
                "optimization_strategy": "clarity"
            })
        else:
            return "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
    
    def count_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def get_stats(self):
        return {
            "model": self.model_name,
            "total_tokens": 0,
            "total_cost": 0.0,
            "calls": self.call_count
        }


def run_offline_test():
    """运行离线测试"""
    logger.info("Starting offline test (no API calls)...")
    
    # 加载数据
    data_dir = PROJECT_ROOT / "data" / "raw"
    users_df = pd.read_csv(data_dir / "users.csv")
    contents_df = pd.read_csv(data_dir / "contents.csv")
    interactions_df = pd.DataFrame(columns=["user_id", "content_id", "interaction_count"])
    
    # 创建 stub agents
    stub_model = StubModelWrapper()
    
    # 创建 agents
    reco_agent = RecommendationAgent(
        contents_df=contents_df,
        interactions_df=interactions_df,
        model_wrapper=stub_model,
        prompt_template="Select {k} stories matching user interests."
    )
    
    eval_agent = EvaluationAgent(
        users_df=users_df,
        contents_df=contents_df,
        recommendation_agent=reco_agent,
        model_wrapper=stub_model,
        use_llm_for_ground_truth=False,
        use_llm_for_tag_simulation=False
    )
    
    optimizer = PromptOptimizerAgent(
        model_wrapper=stub_model
    )
    
    # 运行一个简单的测试循环
    logger.info("Running test cycle...")
    
    # 测试推荐
    user = users_df.iloc[0].to_dict()
    logger.info(f"Testing with user {user['user_id']}")
    
    # 评估
    result = eval_agent.evaluate(user)
    logger.info(f"Evaluation result: P@10={result.precision_at_k:.3f}, R@10={result.recall_at_k:.3f}")
    
    # 测试优化
    from src.agents.prompt_optimizer import OptimizeInput
    opt_result = optimizer.optimize(OptimizeInput(
        eval_history=[{
            "precision_at_k": result.precision_at_k,
            "recall_at_k": result.recall_at_k,
            "simulated_tags": result.simulated_tags,
            "recommended": result.recommended,
            "ground_truth": result.ground_truth
        }],
        current_prompt=reco_agent.prompt_template,
        min_delta=0.01
    ))
    
    logger.info(f"Optimization result: new_prompt={'Yes' if opt_result.new_prompt else 'No'}")
    
    # 显示统计
    logger.info(f"\nModel calls: {stub_model.call_count}")
    logger.info("✅ Offline test completed successfully!")
    
    return True


if __name__ == "__main__":
    # 运行离线测试
    success = run_offline_test()
    
    if success:
        logger.info("\n" + "="*60)
        logger.info("离线测试成功！现在可以安全地运行完整版本。")
        logger.info("建议使用以下命令来控制成本：")
        logger.info("  1. 不使用 LLM 评估: python -m src.orchestrator --no-llm-eval")
        logger.info("  2. 减少样本数: python -m src.orchestrator --sample-users 1")
        logger.info("  3. 减少循环数: python -m src.orchestrator --cycles 1")
        logger.info("="*60)