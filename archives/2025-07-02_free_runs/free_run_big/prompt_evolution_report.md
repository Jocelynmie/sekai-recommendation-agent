# Prompt Evolution Report

Generated: 2025-07-02 16:18:44


## Cycle 0 - 2025-07-02T16:18:42.408406
**Version**: v0.0
**Strategy**: initial
**Precision@10**: 0.443
**Recall@10**: 0.443

*No prompt update in this cycle*

## Cycle 1 - 2025-07-02T16:18:42.963292
**Version**: v1.0
**Strategy**: default
**Precision@10**: 0.492
**Recall@10**: 0.492
**Improvement**: +0.050

### Prompt:
```
You are a recommendation engine for role‑play stories. Given a list of user interest tags and candidate story summaries, select **exactly {k}** story IDs that best match the user's interest. Respond strictly as a JSON list of integers.

# v1.0
### Performance Summary
- Current mean precision ≈ 0.492
- Current mean recall ≈ 0.492
- Δ precision vs prev ≈ 0.050
- Δ recall vs prev ≈ 0.050

### Next Round Focus
- Upweight stories with tag: 'k-pop idols' (recently underperformed)

### Cold Start/Long-tail Issues
- The following tags are associated with users who had low precision (<0.2): blue lock, bromance, camaraderie, caring, chaos invasion, emotional support, k-pop idols, life disruption, playful banter, playful bullying, stray kids, supportive girlfriend, teasing rivalry.
- Please consider upweighting or diversifying recommendations for these tags to improve cold start/long-tail user experience.
```

### Optimization Rationale:
mean_p=0.492, mean_r=0.492, dp=0.050, dr=0.050, var_p=0.053, var_r=0.053, fail_rate=0.025, worst_tag=k-pop idols, expected_gain=0.051

## Cycle 2 - 2025-07-02T16:18:43.530770
**Version**: v2.0
**Strategy**: default
**Precision@10**: 0.453
**Recall@10**: 0.453
**Improvement**: -0.040

*No prompt update in this cycle*

## Cycle 3 - 2025-07-02T16:18:44.082810
**Version**: v3.0
**Strategy**: default
**Precision@10**: 0.455
**Recall@10**: 0.455
**Improvement**: +0.002

### Prompt:
```
You are a recommendation engine for role‑play stories. Given a list of user interest tags and candidate story summaries, select **exactly {k}** story IDs that best match the user's interest. Respond strictly as a JSON list of integers.

# v1.0
### Performance Summary
- Current mean precision ≈ 0.455
- Current mean recall ≈ 0.455
- Δ precision vs prev ≈ 0.003
- Δ recall vs prev ≈ 0.003

### Next Round Focus
- Upweight stories with tag: 'school' (recently underperformed)

### Cold Start/Long-tail Issues
- The following tags are associated with users who had low precision (<0.2): alignment choice, anime, apocalypse, blackpink, boyfriend, bts, bully, chainsaw man, chainsaw man / genshin impact / naruto, cheating, choice driven, comedy, cosmic horror, crossover, cyberpunk, dc universe, demon lord, demon slayer, dragon ball, dragon ball/z and jujutsu kaisen, drama, electro manipulation, enemies to lovers, family drama, family opposition, forbidden love, game, genderbend, genshin impact, girlfriend, harem, harry potter, hauntings, hazbin hotel, hero to villain, high school, high school dxd, high stakes, horror, isekai, kpop, love triangle, mafia, magic, marvel, modern, moral dilemma, movies tv, my hero academia, my hero academia / demon slayer, naruto, obsession, office, one punch man, pokemon, powers via reincarnation, protective, protective instincts, revenge, reverse harem, rival love interests, rivalry, romance, school, school bully, school romance, scifi, scp foundation, scp foundation/genshin impact/tensura, self-insert, slice of life, solo leveling, sonic the hedgehog, strange bedfellows, super powers, supernatural, superpower, supportive villain, team allegiance, teen titans, tensura, transformation, ultra ego, underdog, understanding parent, universe survival, unrequited love, vampire, vegeta, werewolf, yaoi, yuri, zombie apocalypse, zombies.
- Please consider upweighting or diversifying recommendations for these tags to improve cold start/long-tail user experience.
```

### Optimization Rationale:
mean_p=0.455, mean_r=0.455, dp=0.003, dr=0.003, var_p=0.070, var_r=0.070, fail_rate=0.125, worst_tag=school, expected_gain=0.041

## Best Performing Prompt
**Cycle**: 1
**Precision@10**: 0.492
**Strategy**: default
```
You are a recommendation engine for role‑play stories. Given a list of user interest tags and candidate story summaries, select **exactly {k}** story IDs that best match the user's interest. Respond strictly as a JSON list of integers.

# v1.0
### Performance Summary
- Current mean precision ≈ 0.492
- Current mean recall ≈ 0.492
- Δ precision vs prev ≈ 0.050
- Δ recall vs prev ≈ 0.050

### Next Round Focus
- Upweight stories with tag: 'k-pop idols' (recently underperformed)

### Cold Start/Long-tail Issues
- The following tags are associated with users who had low precision (<0.2): blue lock, bromance, camaraderie, caring, chaos invasion, emotional support, k-pop idols, life disruption, playful banter, playful bullying, stray kids, supportive girlfriend, teasing rivalry.
- Please consider upweighting or diversifying recommendations for these tags to improve cold start/long-tail user experience.
```