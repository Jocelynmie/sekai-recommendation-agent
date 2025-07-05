"""
模型包装器 - 根据架构设计更新
支持 Gemini, OpenAI, Anthropic
"""
import os
import time
import random
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from loguru import logger
import json
import hashlib

# 导入预算监控
try:
    from .budget_monitor import get_budget_monitor
except ImportError:
    # 如果预算监控模块不可用，创建一个简单的替代
    def get_budget_monitor():
        class DummyMonitor:
            def add_cost(self, cost, model_name="unknown"):
                return True
        return DummyMonitor()

# 加载环境变量
load_dotenv()

# 导入Google Gemini
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Gemini SDK 未安装")

# 导入OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI SDK 未安装: pip install openai")

# 导入Anthropic
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic SDK 未安装: pip install anthropic")


class BaseModelWrapper(ABC):
    """基础模型包装器接口"""
    
    def __init__(self, model_name: str, temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        self._cache = {}
        self.total_tokens_used = 0
        self.total_cost = 0.0
        
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """生成响应"""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """计算token数"""
        pass
    
    def get_cache_key(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """生成缓存键"""
        content = f"{system_prompt or ''}{prompt}{self.temperature}{self.model_name}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取使用统计"""
        return {
            "model": self.model_name,
            "total_tokens": self.total_tokens_used,
            "total_cost": self.total_cost,
            "cache_size": len(self._cache)
        }
    
    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()
        logger.debug(f"缓存已清除，原大小: {len(self._cache)}")


# 模型注册表和注册装饰器
MODEL_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def create_model(model_type: str = "gemini", **kwargs):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_REGISTRY[model_type](**kwargs)


@register_model("gemini")
class GeminiWrapper(BaseModelWrapper):
    """Gemini 模型包装器"""
    
    MODELS = {
        "flash": "gemini-2.0-flash-exp",  # 推荐Agent使用
        "flash-thinking": "gemini-2.0-flash-thinking-exp-1219",  # 思考模型
        "pro": "gemini-1.5-pro-002",
        "pro-latest": "gemini-exp-1206",
        "pro-2.5": "gemini-2.5-pro",  # 最新Gemini 2.5 Pro
    }
    
    def __init__(self, model_type: str = "flash", temperature: float = 0.7):
        model_name = self.MODELS.get(model_type, self.MODELS["flash"])
        super().__init__(model_name, temperature)
        
        if not GEMINI_AVAILABLE:
            raise ImportError("请安装 google-generativeai: pip install google-generativeai")
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("请设置 GOOGLE_API_KEY 环境变量")
        
        genai.configure(api_key=api_key)
        
        self.generation_config = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4096,
        }
        
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )
        
        logger.info(f"初始化 Gemini 模型: {self.model_name}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        cache_key = self.get_cache_key(prompt, system_prompt)
        if cache_key in self._cache:
            logger.debug("使用缓存响应")
            return self._cache[cache_key]
        
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        try:
            response = self.model.generate_content(full_prompt)
            if response.text:
                result = response.text.strip()
                self._cache[cache_key] = result
                
                # 更新统计
                tokens = self.count_tokens(full_prompt) + self.count_tokens(result)
                self.total_tokens_used += tokens
                
                return result
        except Exception as e:
            logger.error(f"Gemini 生成失败: {e}")
            raise
        
        return ""
    
    def count_tokens(self, text: str) -> int:
        try:
            return self.model.count_tokens(text).total_tokens
        except:
            return len(text) // 4


# class OpenAIWrapper(BaseModelWrapper):
#     """OpenAI 模型包装器"""
    
#     MODELS = {
#         "gpt-4o": "gpt-4o-2024-11-20",  # 最新GPT-4o
#         "gpt-4o-mini": "gpt-4o-mini",  # 快速便宜版本
#         "gpt-4-turbo": "gpt-4-turbo-2024-04-09",  # GPT-4 Turbo
#         "gpt-4-128k": "gpt-4-1106-preview",  # 128k上下文
#     }
    
#     # 价格（每1K tokens）
#     PRICING = {
#         "gpt-4o-2024-11-20": {"input": 0.0025, "output": 0.01},
#         "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
#         "gpt-4-turbo-2024-04-09": {"input": 0.01, "output": 0.03},
#         "gpt-4-1106-preview": {"input": 0.01, "output": 0.03},
#     }
@register_model("openai")
class OpenAIWrapper(BaseModelWrapper):
    """OpenAI 模型包装器"""
    
    MODELS = {
        "gpt-4o": "gpt-4o",  # 修正：使用正确的模型名
        "gpt-4o-mini": "gpt-4o-mini",  # 快速便宜版本
        "gpt-4-turbo": "gpt-4-turbo",  # GPT-4 Turbo
        "gpt-4": "gpt-4",  # 标准 GPT-4
        "gpt-3.5-turbo": "gpt-3.5-turbo",  # GPT-3.5
    }
    
    # 价格（每1K tokens）
    PRICING = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }
    
    def __init__(self, model_type: str = "gpt-4o", temperature: float = 0.7):
        model_name = self.MODELS.get(model_type, model_type)
        super().__init__(model_name, temperature)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("请安装 openai: pip install openai")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")
        
        self.client = OpenAI(api_key=api_key)
        
        # 速率限制配置
        self.max_requests_per_minute = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))
        self.rate_limit_retry_attempts = int(os.getenv("RATE_LIMIT_RETRY_ATTEMPTS", "3"))
        self.request_timestamps = []
        
        logger.info(f"初始化 OpenAI 模型: {self.model_name} (速率限制: {self.max_requests_per_minute}/分钟)")
    
    def _check_rate_limit(self):
        """检查速率限制"""
        current_time = time.time()
        # 清理超过1分钟的时间戳
        self.request_timestamps = [ts for ts in self.request_timestamps if current_time - ts < 60]
        
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            oldest_timestamp = min(self.request_timestamps)
            wait_time = 60 - (current_time - oldest_timestamp) + 1
            logger.warning(f"速率限制: 已达到 {self.max_requests_per_minute}/分钟，等待 {wait_time:.1f}秒")
            time.sleep(wait_time)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        cache_key = self.get_cache_key(prompt, system_prompt)
        if cache_key in self._cache:
            logger.debug("使用缓存响应")
            return self._cache[cache_key]
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # 速率限制检查
        self._check_rate_limit()
        
        # 重试逻辑
        for attempt in range(self.rate_limit_retry_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=4096
                )
                
                # 记录请求时间戳
                self.request_timestamps.append(time.time())
                break
                
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < self.rate_limit_retry_attempts:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  # 指数退避
                    logger.warning(f"速率限制，等待 {wait_time:.1f}秒后重试 (尝试 {attempt + 1}/{self.rate_limit_retry_attempts})")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"OpenAI 生成失败: {e}")
                    raise
        
        result = response.choices[0].message.content.strip()
        self._cache[cache_key] = result
        
        # 更新统计
        if response.usage:
            self.total_tokens_used += response.usage.total_tokens
            
            # 计算成本
            if self.model_name in self.PRICING:
                pricing = self.PRICING[self.model_name]
                input_cost = (response.usage.prompt_tokens / 1000) * pricing["input"]
                output_cost = (response.usage.completion_tokens / 1000) * pricing["output"]
                total_cost = input_cost + output_cost
                self.total_cost += total_cost
                
                # 详细token日志
                if os.getenv("LOG_TOKENS", "true").lower() == "true":
                    logger.info(f"📊 {self.model_name} Token使用: "
                              f"prompt={response.usage.prompt_tokens}, "
                              f"completion={response.usage.completion_tokens}, "
                              f"total={response.usage.total_tokens}, "
                              f"cost=${total_cost:.6f}")
                
                # 预算监控
                budget_monitor = get_budget_monitor()
                if not budget_monitor.add_cost(total_cost, self.model_name):
                    raise RuntimeError("预算超限，停止执行")
        
        return result
    
    def count_tokens(self, text: str) -> int:
        # 使用tiktoken进行精确计算，这里简化处理
        return len(text) // 4


class AnthropicWrapper(BaseModelWrapper):
    """Anthropic Claude 模型包装器"""
    
    MODELS = {
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",  # 最强
        "claude-3.5-haiku": "claude-3-5-haiku-20241022",  # 快速便宜
        "claude-3-opus": "claude-3-opus-20240229",  # 之前的最强
    }
    
    # 价格（每1K tokens）
    PRICING = {
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        "claude-3-5-haiku-20241022": {"input": 0.00025, "output": 0.00125},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    }
    
    def __init__(self, model_type: str = "claude-3.5-sonnet", temperature: float = 0.7):
        model_name = self.MODELS.get(model_type, model_type)
        super().__init__(model_name, temperature)
        
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("请安装 anthropic: pip install anthropic")
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("请设置 ANTHROPIC_API_KEY 环境变量")
        
        self.client = Anthropic(api_key=api_key)
        logger.info(f"初始化 Anthropic 模型: {self.model_name}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        cache_key = self.get_cache_key(prompt, system_prompt)
        if cache_key in self._cache:
            logger.debug("使用缓存响应")
            return self._cache[cache_key]
        
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                temperature=self.temperature,
                system=system_prompt if system_prompt else "You are a helpful assistant.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            result = message.content[0].text.strip()    
            self._cache[cache_key] = result
            
            # 更新统计
            if hasattr(message, 'usage'):
                input_tokens = message.usage.input_tokens
                output_tokens = message.usage.output_tokens
                self.total_tokens_used += input_tokens + output_tokens
                
                # 计算成本
                if self.model_name in self.PRICING:
                    pricing = self.PRICING[self.model_name]
                    input_cost = (input_tokens / 1000) * pricing["input"]
                    output_cost = (output_tokens / 1000) * pricing["output"]
                    self.total_cost += input_cost + output_cost
            
            return result
            
        except Exception as e:
            logger.error(f"Anthropic 生成失败: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        # Claude的token计算近似
        return len(text) // 3


# ===== 根据架构图创建特定的模型实例 =====

def create_recommendation_agent() -> BaseModelWrapper:
    """
    创建推荐智能体
    根据架构：Gemini 2.0 Flash 或 GPT-4o Mini
    """
    # 调试环境变量
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    logger.info(f"Environment check - OpenAI: {'✅' if openai_key else '❌'}, Google: {'✅' if google_key else '❌'}")
    
    # 优先使用 Gemini Flash（免费）
    if google_key:
        try:
            logger.info("使用 Gemini 2.0 Flash 作为推荐模型")
            return GeminiWrapper(model_type="flash", temperature=0.7)
        except Exception as e:
            logger.warning(f"Gemini Flash 不可用: {e}")
    
    # 备选：GPT-4o Mini（便宜）
    if openai_key:
        try:
            logger.info("使用 GPT-4o Mini 作为推荐模型")
            return OpenAIWrapper(model_type="gpt-4o-mini", temperature=0.7)
        except Exception as e:
            logger.warning(f"GPT-4o Mini 不可用: {e}")
    
    # 详细错误信息
    error_msg = "没有可用的推荐模型 - "
    if not openai_key and not google_key:
        error_msg += "OPENAI_API_KEY 和 GOOGLE_API_KEY 都未设置"
    elif not openai_key:
        error_msg += "OPENAI_API_KEY 未设置"
    elif not google_key:
        error_msg += "GOOGLE_API_KEY 未设置"
    
    raise RuntimeError(error_msg)


# def create_evaluation_agent() -> BaseModelWrapper:
#     """
#     创建评估智能体
#     根据架构：GPT-4o (128k) 或 Claude 3.5 Haiku
#     """
#     # 优先使用 GPT-4o（强大的推理能力）
#     if os.getenv("OPENAI_API_KEY"):
#         try:
#             logger.info("使用 GPT-4o (128k) 作为评估模型")
#             return OpenAIWrapper(model_type="gpt-4-128k", temperature=0.3)
#         except Exception as e:
#             logger.warning(f"GPT-4o 不可用: {e}")
    
#     # 备选：Claude 3.5 Haiku（便宜且快速）
#     if os.getenv("ANTHROPIC_API_KEY"):
#         try:
#             logger.info("使用 Claude 3.5 Haiku 作为评估模型")
#             return AnthropicWrapper(model_type="claude-3.5-haiku", temperature=0.3)
#         except Exception as e:
#             logger.warning(f"Claude 3.5 Haiku 不可用: {e}")
    
#     # 最后选择：使用标准 GPT-4o
#     if os.getenv("OPENAI_API_KEY"):
#         try:
#             logger.info("使用标准 GPT-4o 作为评估模型")
#             return OpenAIWrapper(model_type="gpt-4o", temperature=0.3)
#         except Exception as e:
#             logger.warning(f"GPT-4o 不可用: {e}")
    
#     raise RuntimeError("没有可用的评估模型")
def create_evaluation_agent() -> BaseModelWrapper:
    """
    创建评估智能体
    使用可用的最佳模型 - 优先使用最新模型
    """
    # 优先使用 Gemini 2.5 Pro（最新）
    if os.getenv("GOOGLE_API_KEY"):
        try:
            logger.info("使用 Gemini 2.5 Pro 作为评估模型")
            return GeminiWrapper(model_type="pro-2.5", temperature=0.3)
        except Exception as e:
            logger.warning(f"Gemini 2.5 Pro 不可用: {e}")
    
    # 备选：GPT-4o
    if os.getenv("OPENAI_API_KEY"):
        try:
            logger.info("使用 GPT-4o 作为评估模型")
            return OpenAIWrapper(model_type="gpt-4o", temperature=0.3)
        except Exception as e:
            logger.warning(f"GPT-4o 不可用: {e}")
    
    # 备选：使用标准 GPT-4
    if os.getenv("OPENAI_API_KEY"):
        try:
            logger.info("使用 GPT-4 作为评估模型")
            return OpenAIWrapper(model_type="gpt-4", temperature=0.3)
        except Exception as e:
            logger.warning(f"GPT-4 不可用: {e}")
    
    # 备选：Claude 3.5 Haiku
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            logger.info("使用 Claude 3.5 Haiku 作为评估模型")
            return AnthropicWrapper(model_type="claude-3.5-haiku", temperature=0.3)
        except Exception as e:
            logger.warning(f"Claude 3.5 Haiku 不可用: {e}")
    
    # 最后选择：使用 GPT-3.5-turbo
    if os.getenv("OPENAI_API_KEY"):
        try:
            logger.info("使用 GPT-3.5-turbo 作为评估模型")
            return OpenAIWrapper(model_type="gpt-3.5-turbo", temperature=0.3)
        except Exception as e:
            logger.warning(f"GPT-3.5-turbo 不可用: {e}")
    
    raise RuntimeError("没有可用的评估模型")


def create_optimizer_agent() -> BaseModelWrapper:
    """
    创建优化智能体
    根据架构：优先使用最新模型
    """
    # 优先使用 Claude 3.5 Sonnet（创意/分析强，成本低40%）
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            logger.info("使用 Claude 3.5 Sonnet 作为优化模型")
            return AnthropicWrapper(model_type="claude-3.5-sonnet", temperature=0.5)
        except Exception as e:
            logger.warning(f"Claude 3.5 Sonnet 不可用: {e}")
    
    # 备选：Gemini 2.5 Pro（最新）
    if os.getenv("GOOGLE_API_KEY"):
        try:
            logger.info("使用 Gemini 2.5 Pro 作为优化模型")
            return GeminiWrapper(model_type="pro-2.5", temperature=0.5)
        except Exception as e:
            logger.warning(f"Gemini 2.5 Pro 不可用: {e}")
    
    # 备选：GPT-4o
    if os.getenv("OPENAI_API_KEY"):
        try:
            logger.info("使用 GPT-4o 作为优化模型")
            return OpenAIWrapper(model_type="gpt-4o", temperature=0.5)
        except Exception as e:
            logger.warning(f"GPT-4o 不可用: {e}")
    
    # 最后选择：Gemini Flash Thinking
    if os.getenv("GOOGLE_API_KEY"):
        try:
            logger.info("使用 Gemini Flash Thinking 作为优化模型")
            return GeminiWrapper(model_type="flash-thinking", temperature=0.5)
        except Exception as e:
            logger.warning(f"Gemini Flash Thinking 不可用: {e}")
    
    raise RuntimeError("没有可用的优化模型")


# 辅助函数：打印成本统计
def print_cost_summary(agents: Dict[str, BaseModelWrapper]):
    """打印所有agent的成本统计"""
    total_cost = 0
    total_tokens = 0
    
    logger.info("\n" + "="*50)
    logger.info("📊 成本统计")
    logger.info("="*50)
    
    for name, agent in agents.items():
        stats = agent.get_stats()
        logger.info(f"\n{name}:")
        logger.info(f"  模型: {stats['model']}")
        logger.info(f"  Tokens: {stats['total_tokens']:,}")
        logger.info(f"  成本: ${stats['total_cost']:.4f}")
        
        total_cost += stats['total_cost']
        total_tokens += stats['total_tokens']
    
    logger.info(f"\n总计:")
    logger.info(f"  总Tokens: {total_tokens:,}")
    logger.info(f"  总成本: ${total_cost:.4f}")
    logger.info("="*50)


if __name__ == "__main__":
    # 测试代码
    logger.info("测试模型配置...")
    
    # 检查API密钥
    keys = {
        "Google": bool(os.getenv("GOOGLE_API_KEY")),
        "OpenAI": bool(os.getenv("OPENAI_API_KEY")),
        "Anthropic": bool(os.getenv("ANTHROPIC_API_KEY"))
    }
    
    logger.info("API密钥状态:")
    for provider, available in keys.items():
        logger.info(f"  {provider}: {'✅' if available else '❌'}")
    
    # 测试创建agents
    agents = {}
    
    try:
        agents["推荐Agent"] = create_recommendation_agent()
        logger.success("✅ 推荐Agent创建成功")
    except Exception as e:
        logger.error(f"❌ 推荐Agent创建失败: {e}")
    
    try:
        agents["评估Agent"] = create_evaluation_agent()
        logger.success("✅ 评估Agent创建成功")
    except Exception as e:
        logger.error(f"❌ 评估Agent创建失败: {e}")
    
    try:
        agents["优化Agent"] = create_optimizer_agent()
        logger.success("✅ 优化Agent创建成功")
    except Exception as e:
        logger.error(f"❌ 优化Agent创建失败: {e}")
    
    # 简单测试
    if agents:
        logger.info("\n测试生成...")
        for name, agent in agents.items():
            try:
                response = agent.generate("Hello, introduce yourself briefly.")
                logger.info(f"{name} 响应: {response[:100]}...")
            except Exception as e:
                logger.error(f"{name} 测试失败: {e}")