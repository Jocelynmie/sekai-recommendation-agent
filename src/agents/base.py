"""
基础智能体类
所有智能体的父类，提供通用功能
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json
import time
from datetime import datetime
from loguru import logger #type: ignore
import os


class BaseAgent(ABC):
    """基础智能体抽象类"""
    
    def __init__(self, name: str, model_wrapper, config: Optional[Dict[str, Any]] = None):
        """
        初始化基础智能体
        
        Args:
            name: 智能体名称
            model_wrapper: 模型包装器实例
            config: 智能体配置
        """
        self.name = name
        self.model = model_wrapper
        self.config = config or {}
        self.history: List[Dict[str, Any]] = []
        self.state: Dict[str, Any] = {}
        
        # 创建日志目录
        self.log_dir = os.path.join("logs", self.name.lower().replace(" ", "_"))
        os.makedirs(self.log_dir, exist_ok=True)
        
        logger.info(f"初始化智能体: {self.name}")
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理输入并返回结果
        
        Args:
            input_data: 输入数据
            
        Returns:
            处理结果
        """
        pass
    
    def log_interaction(self, input_data: Dict[str, Any], output_data: Dict[str, Any]):
        """
        记录交互历史
        
        Args:
            input_data: 输入数据
            output_data: 输出数据
        """
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "input": input_data,
            "output": output_data
        }
        self.history.append(interaction)
        
        # 保存到文件
        log_file = os.path.join(self.log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(interaction, f, ensure_ascii=False, indent=2)
    
    def get_system_prompt(self) -> str:
        """
        获取系统提示词
        
        Returns:
            系统提示词
        """
        return self.config.get("system_prompt", f"你是{self.name}，一个专业的AI助手。")
    
    def update_state(self, key: str, value: Any):
        """
        更新智能体状态
        
        Args:
            key: 状态键
            value: 状态值
        """
        self.state[key] = value
        logger.debug(f"{self.name} 状态更新: {key} = {value}")
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """
        获取智能体状态
        
        Args:
            key: 状态键
            default: 默认值
            
        Returns:
            状态值
        """
        return self.state.get(key, default)
    
    def reset(self):
        """重置智能体状态"""
        self.state.clear()
        self.history.clear()
        logger.info(f"{self.name} 已重置")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        获取性能指标
        
        Returns:
            性能指标字典
        """
        if not self.history:
            return {}
        
        # 计算基本指标
        total_interactions = len(self.history)
        
        # 计算平均处理时间（如果记录了）
        processing_times = []
        for interaction in self.history:
            if "processing_time" in interaction.get("output", {}):
                processing_times.append(interaction["output"]["processing_time"])
        
        metrics = {
            "total_interactions": total_interactions,
            "average_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0,
            "last_interaction": self.history[-1]["timestamp"] if self.history else None
        }
        
        return metrics
    
    def format_prompt(self, template: str, **kwargs) -> str:
        """
        格式化提示模板
        
        Args:
            template: 提示模板
            **kwargs: 模板参数
            
        Returns:
            格式化后的提示
        """
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"提示模板格式化失败: 缺少参数 {e}")
            raise
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(name='{self.name}')>"


class CachedAgent(BaseAgent):
    """带缓存功能的智能体基类"""
    
    def __init__(self, name: str, model_wrapper, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, model_wrapper, config)
        self.cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_cache_key(self, input_data: Dict[str, Any]) -> str:
        """
        生成缓存键
        
        Args:
            input_data: 输入数据
            
        Returns:
            缓存键
        """
        # 将输入数据转换为稳定的字符串键
        return json.dumps(input_data, sort_keys=True)
    
    def check_cache(self, key: str) -> Optional[Any]:
        """
        检查缓存
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的值（如果存在）
        """
        if key in self.cache:
            self.cache_hits += 1
            logger.debug(f"{self.name} 缓存命中: {key[:50]}...")
            return self.cache[key]
        else:
            self.cache_misses += 1
            return None
    
    def update_cache(self, key: str, value: Any):
        """
        更新缓存
        
        Args:
            key: 缓存键
            value: 要缓存的值
        """
        self.cache[key] = value
        logger.debug(f"{self.name} 缓存更新: {key[:50]}...")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info(f"{self.name} 缓存已清空")


if __name__ == "__main__":
    # 测试代码
    from src.models.gemini_wrapper import create_recommendation_agent #type: ignore
    
    class TestAgent(BaseAgent):
        def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            prompt = input_data.get("prompt", "")
            response = self.model.generate(prompt, self.get_system_prompt())
            return {"response": response}
    
    # 创建测试智能体
    model = create_recommendation_agent()
    agent = TestAgent("测试智能体", model, {"system_prompt": "你是一个友好的助手。"})
    
    # 测试处理
    result = agent.process({"prompt": "你好"})
    logger.info(f"测试结果: {result}")
    
    # 显示性能指标
    metrics = agent.get_performance_metrics()
    logger.info(f"性能指标: {metrics}")