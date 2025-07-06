"""
Model Wrapper Module
Provides unified interface for different LLM providers
"""

import os
import time
import random
import hashlib
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import json

# Import budget monitoring
try:
    from .budget_monitor import get_global_budget_monitor
    BUDGET_MONITOR_AVAILABLE = True
except ImportError:
    # If budget monitoring module is unavailable, create a simple replacement
    BUDGET_MONITOR_AVAILABLE = False
    def get_global_budget_monitor():
        return None

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import Google Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Install with: pip install google-generativeai")

# Import OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not installed. Install with: pip install openai")

# Import Anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic not installed. Install with: pip install anthropic")

from loguru import logger


@dataclass
class ModelStats:
    """Model usage statistics"""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    last_request_time: float = 0.0


class BaseModelWrapper(ABC):
    """Base model wrapper interface"""
    
    def __init__(self, model_name: str, temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.stats = ModelStats()
        self.cache = {}
        self.rate_limit_timestamps = []
        
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response"""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens"""
        pass
    
    def generate_cache_key(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate cache key"""
        content = f"{system_prompt or ''}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return asdict(self.stats)
    
    def clear_cache(self):
        """Clear cache"""
        self.cache.clear()
        logger.info(f"Cleared cache for {self.model_name}")


# Model registry and registration decorator
MODEL_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


@register_model("gemini")
class GeminiWrapper(BaseModelWrapper):
    """Gemini model wrapper"""
    
    MODELS = {
        "flash": "gemini-2.0-flash-exp",  # Recommendation Agent use
        "flash-thinking": "gemini-2.0-flash-thinking-exp-1219",  # Thinking model
        "pro": "gemini-1.5-pro-002",
        "pro-2.5": "gemini-2.5-pro",  # Latest Gemini 2.5 Pro
    }
    
    PRICES = {
        "flash": {"input": 0.0, "output": 0.0},  # Free tier
        "flash-thinking": {"input": 0.0, "output": 0.0},  # Free tier
        "pro": {"input": 0.0025, "output": 0.0075},
        "pro-2.5": {"input": 0.004, "output": 0.012},
    }
    
    def __init__(self, model_name: str = "flash", temperature: float = 0.7):
        super().__init__(model_name, temperature)
        
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not available")
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=self.MODELS.get(model_name, model_name),
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=8192,
            )
        )
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using Gemini"""
        cache_key = self.generate_cache_key(prompt, system_prompt)
        
        # Check cache
        if cache_key in self.cache:
            self.stats.cache_hits += 1
            return self.cache[cache_key]
        
        self.stats.cache_misses += 1
        
        # Rate limiting
        self._check_rate_limit()
        
        start_time = time.time()
        
        try:
            # Prepare content
            content_parts = []
            if system_prompt:
                content_parts.append(system_prompt)
            content_parts.append(prompt)
            
            response = self.model.generate_content(content_parts)
            result = response.text
            
            # Update statistics
            self.stats.total_requests += 1
            self.stats.last_request_time = time.time()
            
            # Calculate cost
            input_tokens = self.count_tokens(prompt)
            output_tokens = self.count_tokens(result)
            
            prices = self.PRICES.get(self.model_name, {"input": 0.0, "output": 0.0})
            cost = (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1000
            
            self.stats.total_tokens += input_tokens + output_tokens
            self.stats.total_cost += cost
            
            # Detailed token logging
            logger.debug(f"Gemini {self.model_name}: {input_tokens} input + {output_tokens} output tokens = ${cost:.6f}")
            
            # Budget monitoring
            if BUDGET_MONITOR_AVAILABLE:
                budget_monitor = get_global_budget_monitor()
                if budget_monitor:
                    budget_monitor.record_usage(input_tokens + output_tokens, cost)
            
            # Cache result
            self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken for precise calculation, simplified here"""
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def _check_rate_limit(self):
        """Check rate limiting"""
        current_time = time.time()
        
        # Clean timestamps older than 1 minute
        self.rate_limit_timestamps = [t for t in self.rate_limit_timestamps if current_time - t < 60]
        
        # Check if we're within rate limit (300 requests per minute for free tier)
        if len(self.rate_limit_timestamps) >= 300:
            wait_time = 60 - (current_time - self.rate_limit_timestamps[0])
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
        
        self.rate_limit_timestamps.append(current_time)


@register_model("openai")
class OpenAIWrapper(BaseModelWrapper):
    """OpenAI model wrapper"""
    
    MODELS = {
        "gpt-4o": "gpt-4o",  # Fixed: use correct model name
        "gpt-4o-mini": "gpt-4o-mini",  # Fast and cheap version
        "gpt-4o-128k": "gpt-4o-128k",  # 128k context
        "gpt-4": "gpt-4",  # Standard GPT-4
        "gpt-3.5-turbo": "gpt-3.5-turbo",
    }
    
    # Prices (per 1K tokens)
    PRICES = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o-128k": {"input": 0.005, "output": 0.015},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.7):
        super().__init__(model_name, temperature)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not available")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        self.client = openai.OpenAI(api_key=api_key)
        
        # Rate limiting configuration
        self.max_retries = 3
        self.rate_limit_window = 60  # seconds
        self.rate_limit_requests = 300  # requests per window
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using OpenAI"""
        cache_key = self.generate_cache_key(prompt, system_prompt)
        
        # Check cache
        if cache_key in self.cache:
            self.stats.cache_hits += 1
            return self.cache[cache_key]
        
        self.stats.cache_misses += 1
        
        # Rate limiting check
        self._check_rate_limit()
        
        # Retry logic
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=self.MODELS.get(self.model_name, self.model_name),
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=8192
                )
                
                result = response.choices[0].message.content
                
                # Record request timestamp
                self.rate_limit_timestamps.append(time.time())
                
                # Update statistics
                self.stats.total_requests += 1
                self.stats.last_request_time = time.time()
                
                # Calculate cost
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                
                prices = self.PRICES.get(self.model_name, {"input": 0.0, "output": 0.0})
                cost = (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1000
                
                self.stats.total_tokens += input_tokens + output_tokens
                self.stats.total_cost += cost
                
                # Detailed token logging
                logger.debug(f"OpenAI {self.model_name}: {input_tokens} input + {output_tokens} output tokens = ${cost:.6f}")
                
                # Budget monitoring
                if BUDGET_MONITOR_AVAILABLE:
                    budget_monitor = get_global_budget_monitor()
                    if budget_monitor:
                        budget_monitor.record_usage(input_tokens + output_tokens, cost)
                
                # Cache result
                self.cache[cache_key] = result
                
                return result
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff
                    logger.warning(f"OpenAI API error (attempt {attempt + 1}/{self.max_retries}): {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"OpenAI API failed after {self.max_retries} attempts: {e}")
                    raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken"""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.MODELS.get(self.model_name, "gpt-4"))
            return len(encoding.encode(text))
        except ImportError:
            # Fallback: rough estimation
            return len(text) // 4
    
    def _check_rate_limit(self):
        """Check rate limiting"""
        current_time = time.time()
        
        # Clean timestamps older than rate limit window
        self.rate_limit_timestamps = [t for t in self.rate_limit_timestamps if current_time - t < self.rate_limit_window]
        
        # Check if we're within rate limit
        if len(self.rate_limit_timestamps) >= self.rate_limit_requests:
            wait_time = self.rate_limit_window - (current_time - self.rate_limit_timestamps[0])
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)


@register_model("anthropic")
class AnthropicWrapper(BaseModelWrapper):
    """Anthropic Claude model wrapper"""
    
    MODELS = {
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",  # Strongest
        "claude-3.5-haiku": "claude-3-5-haiku-20241022",  # Fast and cheap
        "claude-3-opus": "claude-3-opus-20240229",  # Previously strongest
    }
    
    # Prices (per 1K tokens)
    PRICES = {
        "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3.5-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
    }
    
    def __init__(self, model_name: str = "claude-3.5-sonnet", temperature: float = 0.7):
        super().__init__(model_name, temperature)
        
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic not available")
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using Claude"""
        cache_key = self.generate_cache_key(prompt, system_prompt)
        
        # Check cache
        if cache_key in self.cache:
            self.stats.cache_hits += 1
            return self.cache[cache_key]
        
        self.stats.cache_misses += 1
        
        try:
            start_time = time.time()
            
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.messages.create(
                model=self.MODELS.get(self.model_name, self.model_name),
                messages=messages,
                temperature=self.temperature,
                max_tokens=8192
            )
            
            result = response.content[0].text
            
            # Update statistics
            self.stats.total_requests += 1
            self.stats.last_request_time = time.time()
            
            # Calculate cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            
            prices = self.PRICES.get(self.model_name, {"input": 0.0, "output": 0.0})
            cost = (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1000
            
            self.stats.total_tokens += input_tokens + output_tokens
            self.stats.total_cost += cost
            
            # Claude's token calculation approximation
            logger.debug(f"Claude {self.model_name}: {input_tokens} input + {output_tokens} output tokens = ${cost:.6f}")
            
            # Budget monitoring
            if BUDGET_MONITOR_AVAILABLE:
                budget_monitor = get_global_budget_monitor()
                if budget_monitor:
                    budget_monitor.record_usage(input_tokens + output_tokens, cost)
            
            # Cache result
            self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using Claude's tokenizer"""
        try:
            return self.client.count_tokens(text)
        except:
            # Fallback: rough estimation
            return len(text) // 4


# ===== Create specific model instances based on architecture diagram =====

def create_recommendation_agent() -> BaseModelWrapper:
    """Create recommendation agent model"""
    # Debug environment variables
    logger.info("Creating recommendation agent model...")
    
    # Priority: use Gemini Flash (free)
    if GEMINI_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
        try:
            logger.info("Using Gemini 2.0 Flash as recommendation model")
            return GeminiWrapper("flash", temperature=0.7)
        except Exception as e:
            logger.warning(f"Gemini Flash unavailable: {e}")
    
    # Alternative: GPT-4o Mini (cheap)
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            logger.info("Using GPT-4o Mini as recommendation model")
            return OpenAIWrapper("gpt-4o-mini", temperature=0.7)
        except Exception as e:
            logger.warning(f"GPT-4o Mini unavailable: {e}")
    
    # Fallback: use stub
    logger.warning("No API keys available, using stub model")
    from .gemini_wrapper import GeminiWrapper as StubWrapper
    return StubWrapper()


def create_evaluation_agent() -> BaseModelWrapper:
    """Create evaluation agent model"""
    # Priority: use Gemini 2.5 Pro (latest)
    if GEMINI_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
        try:
            logger.info("Using Gemini 2.5 Pro as evaluation model")
            return GeminiWrapper("pro-2.5", temperature=0.3)
        except Exception as e:
            logger.warning(f"Gemini 2.5 Pro unavailable: {e}")
    
    # Alternative: GPT-4o
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            logger.info("Using GPT-4o as evaluation model")
            return OpenAIWrapper("gpt-4o", temperature=0.3)
        except Exception as e:
            logger.warning(f"GPT-4o unavailable: {e}")
    
    # Alternative: use standard GPT-4
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            logger.info("Using GPT-4 as evaluation model")
            return OpenAIWrapper("gpt-4", temperature=0.3)
        except Exception as e:
            logger.warning(f"GPT-4 unavailable: {e}")
    
    # Alternative: Claude 3.5 Haiku
    if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
        try:
            logger.info("Using Claude 3.5 Haiku as evaluation model")
            return AnthropicWrapper("claude-3.5-haiku", temperature=0.3)
        except Exception as e:
            logger.warning(f"Claude 3.5 Haiku unavailable: {e}")
    
    # Last choice: use GPT-3.5-turbo
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            logger.info("Using GPT-3.5-turbo as evaluation model")
            return OpenAIWrapper("gpt-3.5-turbo", temperature=0.3)
        except Exception as e:
            logger.warning(f"GPT-3.5-turbo unavailable: {e}")
    
    # Fallback: use stub
    logger.warning("No API keys available, using stub model")
    from .gemini_wrapper import GeminiWrapper as StubWrapper
    return StubWrapper()


def create_optimizer_agent() -> BaseModelWrapper:
    """Create optimizer agent model"""
    # Priority: use Claude 3.5 Sonnet (strong creativity/analysis, 40% cheaper cost)
    if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
        try:
            logger.info("Using Claude 3.5 Sonnet as optimizer model")
            return AnthropicWrapper("claude-3.5-sonnet", temperature=0.5)
        except Exception as e:
            logger.warning(f"Claude 3.5 Sonnet unavailable: {e}")
    
    # Alternative: Gemini 2.5 Pro (latest)
    if GEMINI_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
        try:
            logger.info("Using Gemini 2.5 Pro as optimizer model")
            return GeminiWrapper("pro-2.5", temperature=0.5)
        except Exception as e:
            logger.warning(f"Gemini 2.5 Pro unavailable: {e}")
    
    # Alternative: GPT-4o
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            logger.info("Using GPT-4o as optimizer model")
            return OpenAIWrapper("gpt-4o", temperature=0.5)
        except Exception as e:
            logger.warning(f"GPT-4o unavailable: {e}")
    
    # Last choice: Gemini Flash Thinking
    if GEMINI_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
        try:
            logger.info("Using Gemini Flash Thinking as optimizer model")
            return GeminiWrapper("flash-thinking", temperature=0.5)
        except Exception as e:
            logger.warning(f"Gemini Flash Thinking unavailable: {e}")
    
    # Fallback: use stub
    logger.warning("No API keys available, using stub model")
    from .gemini_wrapper import GeminiWrapper as StubWrapper
    return StubWrapper()


def print_cost_summary():
    """Print cost statistics for all agents"""
    agents = {
        "Recommendation": create_recommendation_agent(),
        "Evaluation": create_evaluation_agent(),
        "Optimizer": create_optimizer_agent(),
    }
    
    print("\n💰 Cost Summary by Agent:")
    print("=" * 50)
    
    for name, agent in agents.items():
        stats = agent.get_usage_stats()
        print(f"{name:12} | Requests: {stats['total_requests']:3d} | "
              f"Tokens: {stats['total_tokens']:6d} | Cost: ${stats['total_cost']:.4f}")
    
    print("=" * 50)


# Test code
if __name__ == "__main__":
    # Check API keys
    print("🔍 Checking API key configuration...")
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Google": os.getenv("GOOGLE_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY")
    }
    
    for provider, key in api_keys.items():
        status = "✅ Configured" if key else "❌ Not configured"
        print(f"{provider:10}: {status}")
    
    # Test creating agents
    print("\n🧪 Testing agent creation...")
    try:
        reco_agent = create_recommendation_agent()
        eval_agent = create_evaluation_agent()
        opt_agent = create_optimizer_agent()
        print("✅ All agents created successfully")
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
    
    # Simple test
    print("\n🧪 Testing model generation...")
    try:
        agent = create_recommendation_agent()
        response = agent.generate("Hello, how are you?")
        print(f"✅ Test response: {response[:50]}...")
    except Exception as e:
        print(f"❌ Test failed: {e}")