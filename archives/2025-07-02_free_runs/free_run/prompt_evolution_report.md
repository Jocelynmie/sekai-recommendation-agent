# Prompt Evolution Report

Generated: 2025-07-02 15:37:00


## Cycle 0 - 2025-07-02T15:36:59.557701
**Version**: v0.0
**Strategy**: initial
**Precision@10**: 0.387
**Recall@10**: 0.387

*No prompt update in this cycle*

## Cycle 1 - 2025-07-02T15:36:59.829666
**Version**: v1.0
**Strategy**: default
**Precision@10**: 0.293
**Recall@10**: 0.293
**Improvement**: -0.093

*No prompt update in this cycle*

## Cycle 2 - 2025-07-02T15:37:00.095543
**Version**: v2.0
**Strategy**: default
**Precision@10**: 0.433
**Recall@10**: 0.433
**Improvement**: +0.140

### Prompt:
```
You are a recommendation engine for role‑play stories. Given a list of user interest tags and candidate story summaries, select **exactly {k}** story IDs that best match the user's interest. Respond strictly as a JSON list of integers.

# v1.0
### Performance Summary
- Current mean precision ≈ 0.433
- Current mean recall ≈ 0.433
- Δ precision vs prev ≈ 0.140
- Δ recall vs prev ≈ 0.140

### Next Round Focus
- Upweight stories with tag: 'boyfriend' (recently underperformed)

### Cold Start/Long-tail Issues
- The following tags are associated with users who had low precision (<0.2): anime, apocalypse, blackpink, boyfriend, bts, bully, chainsaw man, chainsaw man / genshin impact / naruto, cheating, comedy, cosmic horror, crossover, cyberpunk, demon slayer, drama, enemies to lovers, family drama, family opposition, forbidden love, game, genderbend, genshin impact, girlfriend, harem, harry potter, hauntings, hazbin hotel, horror, kpop, love triangle, mafia, magic, marvel, modern, movies tv, my hero academia, my hero academia / demon slayer, naruto, obsession, office, pokemon, protective, revenge, reverse harem, rival love interests, romance, school, school bully, school romance, scifi, scp foundation, scp foundation/genshin impact/tensura, slice of life, solo leveling, strange bedfellows, supernatural, superpower, supportive villain, tensura, underdog, understanding parent, unrequited love, vampire, werewolf, yaoi, yuri, zombies.
- Please consider upweighting or diversifying recommendations for these tags to improve cold start/long-tail user experience.
```

### Optimization Rationale:
mean_p=0.433, mean_r=0.433, dp=0.140, dr=0.140, var_p=0.069, var_r=0.069, fail_rate=0.200, worst_tag=boyfriend, expected_gain=0.152

## Cycle 3 - 2025-07-02T15:37:00.328087
**Version**: v3.0
**Strategy**: default
**Precision@10**: 0.447
**Recall@10**: 0.447
**Improvement**: +0.013

### Prompt:
```
You are a recommendation engine for role‑play stories. Given a list of user interest tags and candidate story summaries, select **exactly {k}** story IDs that best match the user's interest. Respond strictly as a JSON list of integers.

# v1.0
### Performance Summary
- Current mean precision ≈ 0.447
- Current mean recall ≈ 0.447
- Δ precision vs prev ≈ 0.013
- Δ recall vs prev ≈ 0.013

### Next Round Focus
- Upweight stories with tag: 'jujutsu kaisen' (recently underperformed)

### Cold Start/Long-tail Issues
- The following tags are associated with users who had low precision (<0.2): confident girlfriend, devoted partner, female audience, jujutsu kaisen, kugisaki nobara, lead singer, power couple, rockstar romance, romance chatbot, supportive partner, toxic.
- Please consider upweighting or diversifying recommendations for these tags to improve cold start/long-tail user experience.
```

### Optimization Rationale:
mean_p=0.447, mean_r=0.447, dp=0.013, dr=0.013, var_p=0.028, var_r=0.028, fail_rate=0.067, worst_tag=jujutsu kaisen, expected_gain=0.028

## Best Performing Prompt
**Cycle**: 3
**Precision@10**: 0.447
**Strategy**: default
```
You are a recommendation engine for role‑play stories. Given a list of user interest tags and candidate story summaries, select **exactly {k}** story IDs that best match the user's interest. Respond strictly as a JSON list of integers.

# v1.0
### Performance Summary
- Current mean precision ≈ 0.447
- Current mean recall ≈ 0.447
- Δ precision vs prev ≈ 0.013
- Δ recall vs prev ≈ 0.013

### Next Round Focus
- Upweight stories with tag: 'jujutsu kaisen' (recently underperformed)

### Cold Start/Long-tail Issues
- The following tags are associated with users who had low precision (<0.2): confident girlfriend, devoted partner, female audience, jujutsu kaisen, kugisaki nobara, lead singer, power couple, rockstar romance, romance chatbot, supportive partner, toxic.
- Please consider upweighting or diversifying recommendations for these tags to improve cold start/long-tail user experience.
```