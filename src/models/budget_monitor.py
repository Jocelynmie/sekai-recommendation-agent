#!/usr/bin/env python
"""
Budget Monitor
Tracks and controls API usage costs
"""

import os
import time
from typing import Dict, Any, Optional
from loguru import logger
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class BudgetConfig:
    """Budget configuration"""
    total_budget: float = 10.0  # Total budget in USD
    warning_threshold: float = 0.8  # Warning at 80% of budget
    hard_limit: float = 1.0  # Hard stop at 100% of budget


@dataclass
class BudgetStats:
    """Budget statistics"""
    total_tokens: int = 0
    total_cost: float = 0.0
    requests_count: int = 0
    last_reset: float = 0.0


class BudgetMonitor:
    """Budget monitor"""
    
    def __init__(self, config: Optional[BudgetConfig] = None):
        """
        Initialize budget monitor
        
        Args:
            config: Budget configuration, None to use default
        """
        self.config = config or BudgetConfig()
        self.stats = BudgetStats()
        self.stats.last_reset = time.time()
        
        # Create logs directory
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "budget_log.jsonl"
        
        logger.info(f"💰 Budget monitor initialized: Hard limit=${self.config.hard_limit}")
    
    def check_budget(self, estimated_cost: float = 0.0) -> bool:
        """Check if request is within budget"""
        # Check hard limit
        if self.stats.total_cost + estimated_cost > self.config.total_budget * self.config.hard_limit:
            logger.error(f"🚨 Budget exceeded! Current: ${self.stats.total_cost:.4f}, Limit: ${self.config.total_budget * self.config.hard_limit}")
            return False
        
        # Check warning threshold (80%)
        if self.stats.total_cost > self.config.total_budget * self.config.warning_threshold:
            logger.warning(f"⚠️ Budget warning: ${self.stats.total_cost:.4f}/{self.config.total_budget:.2f} USD used")
        
        return True

    def record_usage(self, tokens: int, cost: float):
        """Record API usage"""
        self.stats.total_tokens += tokens
        self.stats.total_cost += cost
        self.stats.requests_count += 1
        
        # Log usage
        self._log_usage(tokens, cost)

    def _log_usage(self, tokens: int, cost: float):
        """Log usage to file"""
        log_entry = {
            "timestamp": time.time(),
            "tokens": tokens,
            "cost": cost,
            "total_tokens": self.stats.total_tokens,
            "total_cost": self.stats.total_cost,
            "requests_count": self.stats.requests_count
        }
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def get_budget_stats(self) -> Dict[str, Any]:
        """Get budget statistics"""
        runtime = time.time() - self.stats.last_reset
        return {
            "total_tokens": self.stats.total_tokens,
            "total_cost": self.stats.total_cost,
            "requests_count": self.stats.requests_count,
            "budget_used": self.stats.total_cost / self.config.total_budget,
            "budget_remaining": self.config.total_budget - self.stats.total_cost,
            "last_reset": self.stats.last_reset,
            "runtime_hours": runtime / 3600,
            "cost_per_hour": self.stats.total_cost / (runtime / 3600) if runtime > 0 else 0
        }

    def print_cost_summary(self):
        """Print budget summary"""
        stats = self.get_budget_stats()
        logger.info("📊 Budget Summary:")
        logger.info(f"   Total Tokens: {stats['total_tokens']:,}")
        logger.info(f"   Total Cost: ${stats['total_cost']:.4f}")
        logger.info(f"   Requests: {stats['requests_count']}")
        logger.info(f"   Budget Used: {stats['budget_used']:.1%}")
        logger.info(f"   Budget Remaining: ${stats['budget_remaining']:.2f}")
        logger.info(f"   Runtime: {stats['runtime_hours']:.2f} hours")
        logger.info(f"   Cost per hour: ${stats['cost_per_hour']:.4f}")

    def reset_budget(self):
        """Reset budget statistics"""
        self.stats = BudgetStats()
        self.stats.last_reset = time.time()
        logger.info("🔄 Budget statistics reset")


# Global budget monitor instance
_global_budget_monitor: Optional[BudgetMonitor] = None


def get_global_budget_monitor() -> BudgetMonitor:
    """Get global budget monitor"""
    global _global_budget_monitor
    if _global_budget_monitor is None:
        _global_budget_monitor = BudgetMonitor()
    return _global_budget_monitor


def reset_global_budget_monitor():
    """Reset global budget monitor"""
    global _global_budget_monitor
    _global_budget_monitor = None 