# Prompt Evolution Report

Generated: 2025-07-02 16:20:53


## Cycle 0 - 2025-07-02T16:20:48.563094
**Version**: v0.0
**Strategy**: initial
**Precision@10**: 0.508
**Recall@10**: 0.508

*No prompt update in this cycle*

## Cycle 1 - 2025-07-02T16:20:50.882219
**Version**: v1.0
**Strategy**: default
**Precision@10**: 0.497
**Recall@10**: 0.497
**Improvement**: -0.010

### Prompt:
```
You are a recommendation engine for role‑play stories. Given a list of user interest tags and candidate story summaries, select **exactly {k}** story IDs that best match the user's interest. Respond strictly as a JSON list of integers.

# v1.0
### Performance Summary
- Current mean precision ≈ 0.497
- Current mean recall ≈ 0.497
- Δ precision vs prev ≈ -0.010
- Δ recall vs prev ≈ -0.010

### Next Round Focus
- Upweight stories with tag: 'naruto' (recently underperformed)

### Cold Start/Long-tail Issues
- The following tags are associated with users who had low precision (<0.2): abuse of power, anime, apocalypse, blackpink, blue lock, boyfriend, bromance, bts, bully, bully and victim, camaraderie, caring, chainsaw man, chaos invasion, cheating, comedy, competition, crossover, cyberpunk, danmachi, demon slayer, drama, eliminating competition, emotional support, enemies to lovers, family drama, female dominance, forbidden love, game, genderbend, genshin impact, girlfriend, harem, harry potter, hauntings, hazbin hotel, hidden admiration, hidden feelings, high school dxd, horror, jujutsu kaisen, k-pop idols, kpop, life disruption, love confession letter, love triangle, mafia, magic, marvel, modern, movies tv, multiple love interests, my hero academia, naruto, obsession, obsessive, office, playful banter, playful bullying, pokemon, protective, revenge, reverse harem, rivalry, romance, ruthless protection, school, school bully, school romance, scifi, self-insert, slice of life, stray kids, supernatural, superpower, supportive girlfriend, teasing rivalry, tsundere, underdog, unhealthy fixation, unrequited love, vampire, werewolf, yakuza, yandere, yaoi, yuri, zombies.
- Please consider upweighting or diversifying recommendations for these tags to improve cold start/long-tail user experience.
```

### Optimization Rationale:
mean_p=0.497, mean_r=0.497, dp=-0.010, dr=-0.010, var_p=0.060, var_r=0.060, fail_rate=0.100, worst_tag=naruto, expected_gain=0.025

## Cycle 2 - 2025-07-02T16:20:53.134045
**Version**: v2.0
**Strategy**: default
**Precision@10**: 0.447
**Recall@10**: 0.447
**Improvement**: -0.050

*No prompt update in this cycle*

## Best Performing Prompt
**Cycle**: 0
**Precision@10**: 0.508
**Strategy**: initial
```
You are a recommendation engine for role‑play stories. Given a list of user interest tags and candidate story summaries, select **exactly {k}** story IDs that best match the user's interest. Respond strictly as a JSON list of integers.
```