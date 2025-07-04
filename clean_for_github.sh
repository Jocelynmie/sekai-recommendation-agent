#!/bin/bash

echo "🧹 清理项目文件，准备推送到GitHub..."

# 删除实验日志（保留最新一个实验和免费版本作为示例）
echo "📁 清理实验日志..."
find logs/ -maxdepth 1 -type d -name "*" | grep -v "precision_boost_3cycles_15user" | grep -v "free_run" | grep -v "logs$" | xargs rm -rf

# 删除缓存
echo "🗂️ 清理缓存..."
rm -rf cache/

# 删除虚拟环境
echo "🐍 清理虚拟环境..."
rm -rf venv/

# 删除系统文件
echo "🖥️ 清理系统文件..."
find . -name ".DS_Store" -delete
find . -name "Thumbs.db" -delete

# 删除临时/测试文件（保留重要脚本）
echo "🧪 清理临时文件..."
rm -f test_multi_view.py
rm -f test_setup.py
rm -f example_usage.py
rm -f explore_data.py
rm -f build_index.py
rm -f requirements_actual.txt

# 删除归档
echo "📦 清理归档..."
rm -rf archives/

# 删除Python缓存
echo "🐍 清理Python缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete

echo "✅ 清理完成！"
echo "📊 当前目录大小:"
du -sh .

echo "🚀 现在可以推送到GitHub了！"
echo "📝 保留的重要文件:"
echo "  - run_experiment.py (主要实验脚本)"
echo "  - run_free.py (免费版本运行脚本)"
echo "  - analyze_summary.py (结果分析)"
echo "  - README.md (项目文档)"
echo "📁 保留的实验日志:"
echo "  - logs/precision_boost_3cycles_15user/ (完整实验示例)"
echo "  - logs/free_run/ (免费版本实验示例)" 