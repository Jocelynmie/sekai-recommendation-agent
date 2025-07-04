"""
测试项目设置和基础组件
"""
from loguru import logger
from src.models.gemini_wrapper import create_recommendation_agent
from src.agents.base import BaseAgent
from typing import Dict, Any


class SimpleTestAgent(BaseAgent):
    """简单的测试智能体"""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("prompt", "")
        response = self.model.generate(prompt, self.get_system_prompt())
        return {"response": response}


def test_basic_setup():
    """测试基本设置"""
    logger.info("开始测试项目设置...")
    
    # 测试模型创建
    logger.info("1. 测试模型创建...")
    try:
        model = create_recommendation_agent()
        logger.success("✓ 模型创建成功")
    except Exception as e:
        logger.error(f"✗ 模型创建失败: {e}")
        return False
    
    # 测试智能体创建
    logger.info("2. 测试智能体创建...")
    try:
        agent = SimpleTestAgent(
            name="测试智能体",
            model_wrapper=model,
            config={"system_prompt": "你是一个友好的测试助手。"}
        )
        logger.success("✓ 智能体创建成功")
    except Exception as e:
        logger.error(f"✗ 智能体创建失败: {e}")
        return False
    
    # 测试智能体处理
    logger.info("3. 测试智能体处理...")
    try:
        result = agent.process({"prompt": "请说'测试成功'"})
        logger.info(f"智能体响应: {result['response']}")
        logger.success("✓ 智能体处理成功")
    except Exception as e:
        logger.error(f"✗ 智能体处理失败: {e}")
        return False
    
    # 测试性能指标
    logger.info("4. 测试性能指标...")
    try:
        metrics = agent.get_performance_metrics()
        logger.info(f"性能指标: {metrics}")
        logger.success("✓ 性能指标获取成功")
    except Exception as e:
        logger.error(f"✗ 性能指标获取失败: {e}")
        return False
    
    logger.success("所有测试通过！项目设置正确。")
    return True


if __name__ == "__main__":
    test_basic_setup()