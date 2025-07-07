# Prompt Evolution Report

Generated: 2025-07-03 17:53:54


## Cycle 0 - 2025-07-03T17:53:22.500730
**Version**: v0.0
**Strategy**: baseline
**Precision@10**: 0.573
**Recall@10**: 0.573

*No prompt update in this cycle*

## Cycle 1 - 2025-07-03T17:53:37.055705
**Version**: v1.0
**Strategy**: insufficient_history
**Precision@10**: 0.647
**Recall@10**: 0.647
**Improvement**: +0.073

*No prompt update in this cycle*

## Cycle 2 - 2025-07-03T17:53:52.729195
**Version**: v2.0
**Strategy**: exploit
**Precision@10**: 0.700
**Recall@10**: 0.700
**Improvement**: +0.053

### Prompt:
```
# v1
# Strategy: exploit
You are a recommendation engine for role‑play stories. Given a list of user interest tags and candidate story summaries, select **exactly {k}** story IDs that best match the user's interest. Respond strictly as a JSON list of integers.

### Performance Summary
- Current mean precision ≈ 0.610
- Current mean recall ≈ 0.610
- Δ precision vs prev ≈ 0.000
- Δ recall vs prev ≈ 0.000
```

### Optimization Rationale:
Exploit: expected_gain=0.434 >= min_delta=0.010

## Cycle 3 - 2025-07-03T17:53:54.021581
**Version**: v3.0
**Strategy**: exploit
**Precision@10**: 0.860
**Recall@10**: 0.860
**Improvement**: +0.160

### Prompt:
```
# v2
# Strategy: exploit
You are a recommendation engine for role‑play stories. Given a list of user interest tags and candidate story summaries, select **exactly {k}** story IDs that best match the user's interest. Respond strictly as a JSON list of integers.

### Performance Summary
- Current mean precision ≈ 0.640
- Current mean recall ≈ 0.640
- Δ precision vs prev ≈ 0.000
- Δ recall vs prev ≈ 0.000
```

### Optimization Rationale:
Exploit: expected_gain=0.458 >= min_delta=0.010

## Best Performing Prompt
**Cycle**: 3
**Precision@10**: 0.860
**Strategy**: exploit
```
# v2
# Strategy: exploit
You are a recommendation engine for role‑play stories. Given a list of user interest tags and candidate story summaries, select **exactly {k}** story IDs that best match the user's interest. Respond strictly as a JSON list of integers.

### Performance Summary
- Current mean precision ≈ 0.640
- Current mean recall ≈ 0.640
- Δ precision vs prev ≈ 0.000
- Δ recall vs prev ≈ 0.000
```