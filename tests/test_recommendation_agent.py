# tests/test_recommendation_agent.py
import sys
import os
from pathlib import Path

# 获取项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger
from src.agents.recommendation_agent import RecommendationAgent, RecommendationRequest

def test_recommendation_agent():
    """测试推荐智能体"""
    logger.info("开始测试推荐智能体...")
    
    # 加载内容数据
    data_path = project_root / 'data' / 'raw' / 'contents.csv'
    contents_df = pd.read_csv(str(data_path))
    logger.info(f"加载了 {len(contents_df)} 个内容")
    
    # 创建推荐智能体
    agent = RecommendationAgent(contents_df)
    
    # 测试用例：不同类型的用户标签
    test_cases = [
        {
            "name": "动漫爱好者",
            "tags": ["naruto", "romance", "adventure", "original character"]
        },
        {
            "name": "浪漫故事爱好者",
            "tags": ["romance", "slice of life", "fluff", "wholesome"]
        },
        {
            "name": "奇幻冒险爱好者",
            "tags": ["isekai", "fantasy", "adventure", "harem"]
        }
    ]
    
    for test_case in test_cases:
        logger.info(f"\n测试用例: {test_case['name']}")
        logger.info(f"用户标签: {test_case['tags']}")
        
        request = RecommendationRequest(
            user_tags=test_case['tags'],
            num_recommendations=10
        )
        
        response = agent.recommend(request)
        
        logger.info(f"推荐的内容ID: {response.content_ids}")
        logger.info(f"推理说明: {response.reasoning}")
        logger.info(f"Prompt版本: {response.prompt_version}")
        
        # 显示推荐的内容标题
        recommended_contents = contents_df[contents_df['content_id'].isin(response.content_ids)]
        logger.info("推荐的内容标题:")
        for _, content in recommended_contents.iterrows():
            logger.info(f"  - {content['title']}")
        
        # 打印分隔线
        logger.info("-" * 80)

def test_single_recommendation():
    """测试单个推荐请求的详细输出"""
    logger.info("测试单个推荐请求...")
    
    # 使用Path对象构建路径
    data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'contents.csv'
    contents_df = pd.read_csv(str(data_path))
    agent = RecommendationAgent(contents_df)
    
    # 特定的测试标签
    test_tags = ["genshin impact", "romance", "adventure"]
    
    request = RecommendationRequest(
        user_tags=test_tags,
        num_recommendations=5  # 只要5个推荐，便于详细查看
    )
    
    response = agent.recommend(request)
    
    # 详细显示每个推荐的内容
    logger.info(f"\n用户标签: {test_tags}")
    logger.info(f"推荐理由: {response.reasoning}")
    logger.info("\n推荐内容详情:")
    
    for content_id in response.content_ids:
        content = contents_df[contents_df['content_id'] == content_id].iloc[0]
        logger.info(f"\n--- 内容ID: {content_id} ---")
        logger.info(f"标题: {content['title']}")
        logger.info(f"简介: {content['intro'][:200]}...")  # 只显示前200字符
        logger.info(f"角色: {content['character_list']}")

if __name__ == "__main__":
    # 设置日志
    log_path = Path(__file__).parent.parent / 'logs' / 'test_recommendation_{time}.log'
    logger.add(str(log_path))
    
    logger.info("="* 80)
    logger.info("开始推荐智能体测试")
    logger.info("="* 80)
    
    # 基础测试
    test_recommendation_agent()
    
    # 详细测试
    logger.info("\n" + "="* 80)
    logger.info("详细推荐测试")
    logger.info("="* 80)
    test_single_recommendation()
    
    logger.info("\n测试完成！")