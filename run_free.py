#!/usr/bin/env python
"""
完全免费运行版本 - 只使用 Gemini（免费）
"""

import os
import sys
from pathlib import Path
import random
import numpy as np

# 强制使用 Gemini，禁用其他 API
os.environ["USE_ONLY_GEMINI"] = "true"
# 临时清空其他 API keys，避免意外收费
original_openai_key = os.environ.get("OPENAI_API_KEY", "")
original_anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
os.environ["OPENAI_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# 修改 model_wrapper 的导入，使用 monkey patch
import src.models.gemini_wrapper as gemini_stub
import src.models.model_wrapper as model_wrapper
from loguru import logger

# Monkey patch: 只在本脚本作用域内强制所有 GeminiWrapper 和工厂函数都用 stub，不影响主流程
def create_gemini_only():
    logger.info("使用 Gemini Flash (stub)")
    return gemini_stub.GeminiWrapper(model_type="flash", temperature=0.7)

model_wrapper.GeminiWrapper = gemini_stub.GeminiWrapper
model_wrapper.create_recommendation_agent = create_gemini_only
model_wrapper.create_evaluation_agent = create_gemini_only
model_wrapper.create_optimizer_agent = create_gemini_only

# 现在运行 orchestrator
if __name__ == "__main__":
    import argparse
    from src.orchestrator import main
    
    parser = argparse.ArgumentParser(description="免费运行 Sekai Orchestrator")
    parser.add_argument("--cycles", type=int, default=4, help="循环次数")
    parser.add_argument("--sample-users", type=int, default=74, help="每轮用户数")
    parser.add_argument("--log-dir", type=Path, 
                       default=PROJECT_ROOT / "logs" / "free_run",
                       help="日志目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("运行免费版本 - 仅使用 Gemini")
    logger.info("="*60)
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    try:
        main(
            cycles=args.cycles,
            min_delta=0.01,
            sample_users=args.sample_users,
            log_dir=args.log_dir,
            use_llm_evaluation=False  # 使用关键词评估
        )
    finally:
        # 恢复原始 API keys
        if original_openai_key:
            os.environ["OPENAI_API_KEY"] = original_openai_key
        if original_anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = original_anthropic_key
    
    logger.info("\n✅ 免费运行完成！")
    logger.info(f"结果保存在: {args.log_dir}")