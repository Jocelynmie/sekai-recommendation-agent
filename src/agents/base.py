"""
Base Agent Classes
Provides foundation for all recommendation system agents
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import hashlib

from src.models.model_wrapper import BaseModelWrapper

logger = logging.getLogger(__name__)


@dataclass
class AgentStats:
    """Agent performance statistics"""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_response_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


class BaseAgent(ABC):
    """Base agent abstract class"""

    def __init__(self, name: str, model_wrapper: BaseModelWrapper, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.model = model_wrapper
        self.config = config or {}
        self.stats = AgentStats()
        
        # Create log directory
        self.log_dir = Path("logs") / name.lower().replace(" ", "_")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        logger.info(f"Initialized {name} with model: {model_wrapper.model_name if hasattr(model_wrapper, 'model_name') else 'unknown'}")

    def _setup_logging(self):
        """Setup agent-specific logging"""
        log_file = self.log_dir / "interactions.jsonl"
        self.log_file = log_file
        
        # Create log file if it doesn't exist
        if not log_file.exists():
            log_file.touch()

    def log_interaction(self, input_data: Dict[str, Any], output_data: Dict[str, Any]):
        """Log agent interaction for analysis"""
        interaction = {
            "timestamp": time.time(),
            "agent": self.name,
            "input": input_data,
            "output": output_data,
            "stats": asdict(self.stats)
        }
        
        # Save to file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(interaction, ensure_ascii=False) + "\n")

    def get_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        return {
            "name": self.name,
            "total_requests": self.stats.total_requests,
            "total_tokens": self.stats.total_tokens,
            "total_cost": self.stats.total_cost,
            "avg_response_time": self.stats.avg_response_time,
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "cache_hit_rate": self.stats.cache_hits / (self.stats.cache_hits + self.stats.cache_misses) if (self.stats.cache_hits + self.stats.cache_misses) > 0 else 0.0
        }

    def reset_stats(self):
        """Reset agent state"""
        self.stats = AgentStats()
        logger.info(f"Reset statistics for {self.name}")

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return results"""
        pass

    def _update_stats(self, tokens: int = 0, cost: float = 0.0, response_time: float = 0.0):
        """Update agent statistics"""
        self.stats.total_requests += 1
        self.stats.total_tokens += tokens
        self.stats.total_cost += cost
        
        # Calculate average response time (if recorded)
        if response_time > 0:
            if self.stats.avg_response_time == 0:
                self.stats.avg_response_time = response_time
            else:
                self.stats.avg_response_time = (self.stats.avg_response_time + response_time) / 2


class CachedAgent(BaseAgent):
    """Agent with caching functionality"""

    def __init__(self, name: str, model_wrapper: BaseModelWrapper, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, model_wrapper, config)
        self.cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}

    def _get_cache_key(self, input_data: Dict[str, Any]) -> str:
        """Convert input data to stable string key"""
        # Sort keys for consistent ordering
        sorted_data = dict(sorted(input_data.items()))
        data_str = json.dumps(sorted_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(data_str.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get result from cache"""
        if cache_key in self.cache:
            self.cache_stats["hits"] += 1
            self.stats.cache_hits += 1
            return self.cache[cache_key]
        else:
            self.cache_stats["misses"] += 1
            self.stats.cache_misses += 1
            return None

    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """Save result to cache"""
        self.cache[cache_key] = result

    def clear_cache(self):
        """Clear cache"""
        self.cache.clear()
        self.cache_stats = {"hits": 0, "misses": 0}
        logger.info(f"Cleared cache for {self.name}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            "cache_size": len(self.cache),
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "hit_rate": hit_rate
        }


# Test code
if __name__ == "__main__":
    # Create test agent
    class TestAgent(BaseAgent):
        def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            return {"result": "test", "input": input_data}

    # Test processing
    agent = TestAgent("TestAgent", None)
    result = agent.process({"test": "data"})
    print(f"Test result: {result}")
    
    # Display performance metrics
    stats = agent.get_stats()
    print(f"Agent stats: {stats}")