#!/usr/bin/env python3
"""
Sekai Recommendation System - One-Command Experiment Runner
===========================================================

This script runs a complete experiment with comprehensive logging and analysis.
It demonstrates the AI-native approach, content understanding, and robust evaluation.

Usage:
    python run_experiment.py [--cycles N] [--users N] [--mode llm|vector]
"""

import argparse
import sys
from pathlib import Path
import subprocess
import json
from datetime import datetime

def run_experiment(cycles=3, users=10, mode="llm", log_dir=None):
    """运行完整的推荐系统实验"""
    
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/experiment_{timestamp}"
    
    print(f"🚀 开始 Sekai 推荐系统实验")
    print(f"📊 参数: {cycles} 轮, {users} 用户/轮, 模式: {mode}")
    print(f"📁 日志目录: {log_dir}")
    
    # 构建命令
    cmd = [
        "python", "-m", "src.main",
        "--cycles", str(cycles),
        "--sample-users", str(users),
        "--recall-mode", mode,
        "--eval-mode", "llm" if mode == "llm" else "keyword",
        "--log-dir", log_dir
    ]
    
    print(f"\n🔧 执行命令: {' '.join(cmd)}")
    
    try:
        # 运行实验
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 实验完成!")
            
            # 分析结果
            analyze_results(log_dir)
            
        else:
            print(f"❌ 实验失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        return False
    
    return True

def analyze_results(log_dir):
    """分析实验结果"""
    print(f"\n📈 分析结果: {log_dir}")
    
    # 读取 summary.json
    summary_path = Path(log_dir) / "summary.json"
    if not summary_path.exists():
        print("❌ 找不到 summary.json")
        return
    
    with open(summary_path, 'r') as f:
        data = json.load(f)
    
    print(f"\n📊 实验结果摘要:")
    print(f"  总轮数: {len(data)}")
    
    if data:
        initial_precision = data[0]['precision_at_k']
        final_precision = data[-1]['precision_at_k']
        improvement = final_precision - initial_precision
        
        print(f"  初始 Precision@10: {initial_precision:.3f}")
        print(f"  最终 Precision@10: {final_precision:.3f}")
        print(f"  改进: {improvement:+.3f}")
        
        if initial_precision > 0:
            print(f"  相对改进: {improvement/initial_precision*100:+.1f}%")
    
    # 显示每轮结果
    print(f"\n📋 详细结果:")
    print("  轮次 | Precision | Recall | 策略")
    print("  -----|-----------|--------|------")
    
    for cycle in data:
        print(f"  {cycle['cycle']:2d}    | {cycle['precision_at_k']:.3f}     | {cycle['recall_at_k']:.3f}   | {cycle.get('optimization_strategy', 'N/A')}")

def main():
    parser = argparse.ArgumentParser(
        description="Sekai Recommendation System Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 快速测试 (1轮, 5用户)
  python run_experiment.py --cycles 1 --users 5
  
  # 完整实验 (3轮, 15用户, LLM模式)
  python run_experiment.py --cycles 3 --users 15 --mode llm
  
  # 向量模式实验 (更快, 成本更低)
  python run_experiment.py --cycles 2 --users 10 --mode vector
        """
    )
    
    parser.add_argument(
        "--cycles", 
        type=int, 
        default=3, 
        help="实验轮数 (default: 3)"
    )
    parser.add_argument(
        "--users", 
        type=int, 
        default=10, 
        help="每轮用户数 (default: 10)"
    )
    parser.add_argument(
        "--mode", 
        choices=["llm", "vector"], 
        default="llm",
        help="推荐模式 (default: llm)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        help="自定义日志目录"
    )
    
    args = parser.parse_args()
    
    # 检查环境
    print("🔍 检查环境...")
    try:
        import pandas as pd
        import numpy as np
        from src.agents.multi_view_recall import MultiViewRecall
        print("✅ 依赖检查通过")
    except ImportError as e:
        print(f"❌ 依赖缺失: {e}")
        print("请运行: pip install -r requirements.txt")
        return
    
    # 运行实验
    success = run_experiment(
        cycles=args.cycles,
        users=args.users,
        mode=args.mode,
        log_dir=args.log_dir
    )
    
    if success:
        print(f"\n🎉 实验完成! 查看结果: {args.log_dir or 'logs/experiment_*'}")
    else:
        print(f"\n❌ 实验失败，请检查错误信息")

if __name__ == "__main__":
    main() 