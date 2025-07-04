import json
import matplotlib.pyplot as plt

# 读取 summary.json
with open('logs/vector_only_test/summary.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

cycles = []
precisions = []
recalls = []
methods = []

for cycle in data:
    cycles.append(cycle['cycle'])
    precisions.append(cycle['precision_at_k'])
    recalls.append(cycle['recall_at_k'])
    # 统计 method_used 分布
    method_set = set()
    for user in cycle['user_results']:
        method_set.add(user.get('method_used', 'unknown'))
    methods.append(','.join(method_set))

print("Cycle\tPrecision\tRecall\tMethod")
for c, p, r, m in zip(cycles, precisions, recalls, methods):
    print(f"{c}\t{p:.3f}\t\t{r:.3f}\t{m}")

# 可视化
plt.figure(figsize=(8,5))
plt.plot(cycles, precisions, marker='o', label='Precision@k')
plt.plot(cycles, recalls, marker='s', label='Recall@k')
plt.xlabel('Cycle')
plt.ylabel('Score')
plt.title('Precision/Recall Trend (Vector Only)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('logs/vector_only_test/score_trend.png')
plt.show() 