#!/usr/bin/env python3
"""
使用Gemini Flash的完整实验运行脚本
确保RecommendationAgent优先使用Gemini 2.0 Flash
"""

import os
import sys
from pathlib import Path
import random
import numpy as np
from datetime import datetime

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 强制设置环境变量，确保使用Gemini
os.environ["FORCE_GEMINI_RECOMMENDATION"] = "true"

from loguru import logger

def check_gemini_priority():
    """检查Gemini是否被正确设置为推荐模型的优先选择"""
    
    logger.info("🔍 检查Gemini优先级设置...")
    
    # 检查环境变量
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    logger.info(f"Google API Key: {'✅' if google_key else '❌'}")
    logger.info(f"OpenAI API Key: {'✅' if openai_key else '❌'}")
    
    if not google_key:
        logger.error("❌ 未设置GOOGLE_API_KEY，无法使用Gemini Flash")
        return False
    
    if google_key == "your_google_api_key_here":
        logger.error("❌ GOOGLE_API_KEY仍为默认值，请设置真实的API密钥")
        return False
    
    logger.info("✅ Gemini优先级检查通过")
    return True

def run_gemini_experiment():
    """运行使用Gemini Flash的完整实验"""
    
    logger.info("🚀 开始Gemini Flash优先实验")
    logger.info("=" * 60)
    
    # 创建时间戳日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "logs" / f"gemini_priority_{timestamp}"
    
    logger.info(f"📁 日志目录: {log_dir}")
    
    # 导入orchestrator
    from src.orchestrator import main
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    try:
        # 运行实验
        main(
            cycles=3,  # 3轮优化
            min_delta=0.01,  # 最小改进阈值
            sample_users=15,  # 每轮15个用户
            log_dir=log_dir,
            use_llm_evaluation=True,  # 使用LLM评估
            dry_run=False,
            recall_mode="llm",  # 使用LLM重排序
            rerank_window=30,
            eval_mode="llm",
            use_simple_rerank=True,
            tag_weight=0.1,
            cold_start_boost=0.2,
        )
        
        logger.success("✅ Gemini优先实验完成！")
        logger.info(f"📊 结果保存在: {log_dir}")
        
        # 显示结果摘要
        summary_file = log_dir / "summary.json"
        if summary_file.exists():
            import json
            with open(summary_file, 'r') as f:
                data = json.load(f)
            
            logger.info("\n📈 实验结果摘要:")
            logger.info(f"总轮数: {len(data)}")
            
            if data:
                initial_precision = data[0]['precision_at_k']
                final_precision = data[-1]['precision_at_k']
                improvement = final_precision - initial_precision
                
                logger.info(f"初始 Precision@10: {initial_precision:.3f}")
                logger.info(f"最终 Precision@10: {final_precision:.3f}")
                logger.info(f"改进: {improvement:+.3f}")
                
                if initial_precision > 0:
                    logger.info(f"相对改进: {improvement/initial_precision*100:+.1f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 实验运行失败: {e}")
        return False

def main():
    """主函数"""
    
    logger.info("🎯 Sekai Recommendation Agent - Gemini优先模式")
    logger.info("=" * 60)
    
    # 检查Gemini优先级
    if not check_gemini_priority():
        logger.error("请先配置正确的GOOGLE_API_KEY")
        return False
    
    # 运行实验
    success = run_gemini_experiment()
    
    if success:
        logger.info("\n🎉 实验成功完成！")
        logger.info("💡 提示: 检查日志文件确认RecommendationAgent使用了Gemini Flash")
    else:
        logger.error("\n❌ 实验失败")
    
    return success

if __name__ == "__main__":
    main() 