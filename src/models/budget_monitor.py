#!/usr/bin/env python
"""
预算监控模块
用于跟踪和控制API使用成本
"""

import os
import time
from typing import Dict, Optional
from loguru import logger


class BudgetMonitor:
    """预算监控器"""
    
    def __init__(self, hard_limit: Optional[float] = None):
        """
        初始化预算监控器
        
        Args:
            hard_limit: 硬限制（美元），None表示从环境变量读取
        """
        self.hard_limit = hard_limit or float(os.getenv("OPENAI_BILLING_HARD_LIMIT", "50.00"))
        self.current_cost = 0.0
        self.start_time = time.time()
        self.cost_history = []
        
        logger.info(f"💰 预算监控初始化: 硬限制=${self.hard_limit}")
    
    def add_cost(self, cost: float, model_name: str = "unknown") -> bool:
        """
        添加成本并检查是否超限
        
        Args:
            cost: 新增成本（美元）
            model_name: 模型名称
            
        Returns:
            bool: True表示可以继续，False表示已超限
        """
        self.current_cost += cost
        self.cost_history.append({
            "timestamp": time.time(),
            "cost": cost,
            "model": model_name,
            "total": self.current_cost
        })
        
        # 检查硬限制
        if self.current_cost >= self.hard_limit:
            logger.error(f"🚨 预算超限! 当前: ${self.current_cost:.4f}, 限制: ${self.hard_limit}")
            return False
        
        # 检查警告阈值（80%）
        warning_threshold = self.hard_limit * 0.8
        if self.current_cost >= warning_threshold:
            logger.warning(f"⚠️ 预算警告: ${self.current_cost:.4f}/{self.hard_limit} "
                          f"({self.current_cost/self.hard_limit*100:.1f}%)")
        else:
            logger.info(f"💰 成本更新: +${cost:.6f} ({model_name}), "
                       f"总计: ${self.current_cost:.4f}/{self.hard_limit}")
        
        return True
    
    def get_stats(self) -> Dict:
        """获取预算统计"""
        runtime = time.time() - self.start_time
        return {
            "current_cost": self.current_cost,
            "hard_limit": self.hard_limit,
            "remaining": self.hard_limit - self.current_cost,
            "usage_percent": (self.current_cost / self.hard_limit) * 100,
            "runtime_hours": runtime / 3600,
            "cost_per_hour": self.current_cost / (runtime / 3600) if runtime > 0 else 0,
            "total_requests": len(self.cost_history)
        }
    
    def print_summary(self):
        """打印预算摘要"""
        stats = self.get_stats()
        logger.info("📊 预算摘要:")
        logger.info(f"  当前成本: ${stats['current_cost']:.4f}")
        logger.info(f"  硬限制: ${stats['hard_limit']:.2f}")
        logger.info(f"  剩余预算: ${stats['remaining']:.4f}")
        logger.info(f"  使用率: {stats['usage_percent']:.1f}%")
        logger.info(f"  运行时间: {stats['runtime_hours']:.2f}小时")
        logger.info(f"  每小时成本: ${stats['cost_per_hour']:.4f}")
        logger.info(f"  总请求数: {stats['total_requests']}")


# 全局预算监控器实例
_global_monitor: Optional[BudgetMonitor] = None


def get_budget_monitor() -> BudgetMonitor:
    """获取全局预算监控器"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = BudgetMonitor()
    return _global_monitor


def reset_budget_monitor():
    """重置全局预算监控器"""
    global _global_monitor
    _global_monitor = None 