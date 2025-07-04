#!/usr/bin/env python
"""
推荐系统使用示例
展示如何配置和使用费用与速率监控
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger


def setup_monitoring():
    """设置监控环境变量"""
    logger.info("🔧 设置监控环境变量...")
    
    # 速率限制配置
    os.environ["MAX_REQUESTS_PER_MINUTE"] = "60"
    os.environ["RATE_LIMIT_RETRY_ATTEMPTS"] = "3"
    
    # 日志配置
    os.environ["LOG_TOKENS"] = "true"
    os.environ["LOG_COSTS"] = "true"
    
    # 预算控制
    os.environ["OPENAI_BILLING_HARD_LIMIT"] = "10.00"  # 10美元限制
    
    logger.info("✅ 监控环境变量设置完成")


def run_with_monitoring():
    """运行带监控的推荐系统"""
    logger.info("🚀 运行带监控的推荐系统...")
    
    try:
        from src.models.budget_monitor import get_budget_monitor
        from src.models.model_wrapper import create_recommendation_agent
        
        # 获取预算监控器
        budget_monitor = get_budget_monitor()
        
        # 创建推荐智能体
        agent = create_recommendation_agent()
        
        # 模拟一些请求
        test_prompts = [
            "推荐一些关于动漫的故事",
            "推荐一些关于浪漫的故事", 
            "推荐一些关于冒险的故事"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            logger.info(f"📝 请求 {i}: {prompt}")
            
            try:
                # 这里只是示例，实际需要真实的API调用
                # response = agent.generate(prompt)
                logger.info(f"✅ 请求 {i} 完成")
                
            except Exception as e:
                logger.error(f"❌ 请求 {i} 失败: {e}")
                break
        
        # 打印预算摘要
        budget_monitor.print_summary()
        
    except Exception as e:
        logger.error(f"❌ 运行失败: {e}")


def main():
    """主函数"""
    logger.info("🎯 推荐系统监控示例")
    
    # 设置监控
    setup_monitoring()
    
    # 运行示例
    run_with_monitoring()
    
    logger.info("🎉 示例运行完成")


if __name__ == "__main__":
    main() 