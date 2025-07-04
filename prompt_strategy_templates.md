# Prompt & Strategy Templates for Recommendation System

## 1. LLM Rerank Prompt Template

```
You are an expert recommendation assistant. Given a user profile and a list of candidate items, rerank the items to maximize user satisfaction.

User profile:
{user_profile}

Candidate items:
{candidate_list}

Return the top {k} items in order of relevance, with a brief justification for each.
```

---

## 2. Tag Upweighting Strategy (for Cold Start/Long-tail)

```
When scoring candidate items, increase the weight for items containing tags that are rare or new for the user.

For each candidate:
- Compute base score (e.g., vector similarity)
- If candidate contains a cold/rare tag, multiply score by {cold_tag_boost}
- Optionally, add a bonus for tag overlap with user's interests
```

---

## 3. Expanded Vector Recall Strategy

```
To improve recall, expand the candidate set by:
- Using multiple user embedding variants (e.g., original, with cold tags, with recent interests)
- For each variant, retrieve top N candidates from the vector index
- Merge and deduplicate candidates before rerank
```

---

## 4. Cold Start Handling Prompt

```
A new user has limited interaction history. Recommend items by:
- Prioritizing items with popular or trending tags
- Using demographic or contextual similarity if available
- Optionally, ask clarifying questions to the user to refine preferences
```

---

## 5. Evaluation Prompt (LLM-based)

```
You are an impartial evaluator. Given a user profile, a ground-truth item, and a recommended item, judge whether the recommendation is relevant and valuable to the user.

User profile:
{user_profile}

Ground-truth item:
{ground_truth}

Recommended item:
{recommendation}

Respond with a score from 1 (not relevant) to 5 (highly relevant), and a brief explanation.
```

---

## 6. General Strategy Notes

- Always log which strategy and prompt are used for each recommendation.
- Tune parameters (e.g., cold_tag_boost) based on offline evaluation results.
- For production, monitor both precision and diversity metrics.
