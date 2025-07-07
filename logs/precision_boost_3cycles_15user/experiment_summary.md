# Sekai Recommendation System - Experiment Summary

Generated: 2025-07-03 17:53:54

## Overall Performance

- **Initial Precision@10**: 0.573
- **Final Precision@10**: 0.860
- **Total Improvement**: +0.287 (+50.0%)
- **Number of Cycles**: 4
- **Sample Size per Cycle**: 15

## Best Performance

- **Cycle**: 3
- **Precision@10**: 0.860
- **Recall@10**: 0.860
- **Optimization Strategy**: exploit

## Detailed Results

| Cycle | Precision@10 | Recall@10 | Strategy | Expected Gain | Actual Gain |
|-------|-------------|-----------|----------|---------------|-------------|
| 0 | 0.573 | 0.573 | baseline | +0.000 | +0.000 |
| 1 | 0.647 | 0.647 | insufficient_history | +0.000 | +0.073 |
| 2 | 0.700 | 0.700 | exploit | +0.434 | +0.053 |
| 3 | 0.860 | 0.860 | exploit | +0.458 | +0.160 |