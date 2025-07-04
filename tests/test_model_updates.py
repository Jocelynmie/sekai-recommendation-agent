"""
测试更新后的模型配置
"""
from loguru import logger
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.gemini_wrapper import (
    create_recommendation_agent,
    create_evaluation_agent,
    create_optimizer_agent,
    GeminiWrapper
)


def test_model_availability():
    """测试各个模型的可用性"""
    logger.info("=== 测试模型可用性 ===")
    
    # 定义要测试的模型
    models_to_test = [
        ("Flash 2.0 (推荐用)", "flash"),
        ("Pro 2.0 实验版", "pro"),
        ("最新实验模型", "pro-latest"),
        ("思考模型", "flash-thinking"),
        ("Pro 稳定版", "pro-stable"),
        ("Flash 稳定版", "flash-stable")
    ]
    
    available_models = []
    
    for model_name, model_type in models_to_test:
        logger.info(f"\n测试 {model_name} ({model_type})...")
        try:
            model = GeminiWrapper(model_type=model_type, temperature=0.3)
            
            # 简单测试
            response = model.generate("回答：1+1=?", "你是一个数学助手。")
            
            if response:
                logger.success(f"✓ {model_name} 可用")
                logger.info(f"  实际模型: {model.model_name}")
                logger.info(f"  响应: {response.strip()}")
                available_models.append((model_name, model_type, model.model_name))
            else:
                logger.warning(f"✗ {model_name} 响应为空")
                
        except Exception as e:
            logger.error(f"✗ {model_name} 不可用: {str(e)}")
    
    return available_models


def test_agent_creation():
    """测试智能体创建函数"""
    logger.info("\n=== 测试智能体创建 ===")
    
    agents = [
        ("推荐智能体", create_recommendation_agent),
        ("评估智能体", create_evaluation_agent),
        ("优化智能体", create_optimizer_agent)
    ]
    
    created_agents = []
    
    for agent_name, creator_func in agents:
        logger.info(f"\n创建 {agent_name}...")
        try:
            agent = creator_func()
            info = agent.get_model_info()
            
            logger.success(f"✓ {agent_name} 创建成功")
            logger.info(f"  模型类型: {info['model_type']}")
            logger.info(f"  实际模型: {info['model_name']}")
            logger.info(f"  温度: {info['temperature']}")
            
            created_agents.append((agent_name, info))
            
        except Exception as e:
            logger.error(f"✗ {agent_name} 创建失败: {str(e)}")
    
    return created_agents


def test_prompt_optimization_capability():
    """测试prompt优化能力"""
    logger.info("\n=== 测试Prompt优化能力 ===")
    
    try:
        optimizer = create_optimizer_agent()
        
        # 测试prompt分析能力
        test_prompt = """
        You are a content recommendation system.
        User tags: {tags}
        Contents: {contents}
        Return 10 IDs.
        """
        
        analysis_prompt = f"""
        分析以下推荐系统的prompt，找出可以改进的地方：
        
        ```
        {test_prompt}
        ```
        
        请从以下角度分析：
        1. 指令清晰度
        2. 输出格式规范
        3. 匹配逻辑说明
        4. 上下文信息利用
        
        给出具体的改进建议。
        """
        
        response = optimizer.generate(analysis_prompt)
        logger.info("Prompt分析结果：")
        logger.info(response[:500] + "..." if len(response) > 500 else response)
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        return False


def main():
    """运行所有测试"""
    logger.info("🚀 开始测试更新后的模型配置")
    logger.info("=" * 60)
    
    # 1. 测试模型可用性
    available_models = test_model_availability()
    
    # 2. 测试智能体创建
    created_agents = test_agent_creation()
    
    # 3. 测试prompt优化能力
    optimization_ok = test_prompt_optimization_capability()
    
    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("📊 测试总结")
    logger.info(f"\n可用模型 ({len(available_models)}):")
    for name, type_key, actual_model in available_models:
        logger.info(f"  - {name}: {actual_model}")
    
    logger.info(f"\n成功创建的智能体 ({len(created_agents)}):")
    for name, info in created_agents:
        logger.info(f"  - {name}: {info['model_name']}")
    
    logger.info(f"\nPrompt优化能力: {'✓ 正常' if optimization_ok else '✗ 异常'}")
    
    # 建议
    logger.info("\n💡 建议:")
    if len(available_models) < 3:
        logger.warning("可用模型较少，建议检查API配额或网络连接")
    else:
        logger.success("模型配置正常，可以继续开发")


if __name__ == "__main__":
    main()