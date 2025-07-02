"""
Recommendation Agent
────────────────────
• 接收用户勾选的兴趣 `tags`，返回 K 个最相关的 `content_id`
• 首要指标：**速度**（默认使用 Gemini‑2 Flash 或同等速模型）

功能要点
1. system_prompt / user_prompt 拆分，节省 token
2. token‑budget 自适应裁剪内容摘要，确保 prompt 不超长
3. 关键词命中 + 历史交互次数多信号打分
4. Prompt 模板热更新，供 Optimizer 调用
5. 兼容旧测试：RecommendationRequest / recommend()
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger
import numpy as np

# ------------------------------------------------------------------ #
#              让 `pytest` 直接运行时也能找到 src 包                 #
# ------------------------------------------------------------------ #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.base import BaseAgent
from src.models.model_wrapper import create_recommendation_agent

# ===================== 兼容层 dataclass（供旧测试导入） ===================== #
@dataclass
class RecommendationRequest:
    user_tags: List[str]
    num_recommendations: int = 10


@dataclass
class RecommendationResponse:
    content_ids: List[int]
    reasoning: str = ""
    prompt_version: str = ""
    model_used: str = ""


# =========================== 内部数据结构 =========================== #
@dataclass
class Candidate:
    content_id: int
    score: float
    summary: str = field(repr=False)


@dataclass
class RecoResult:
    content_ids: List[int]
    prompt_tokens: int
    model_tokens: int
    model_used: str
    reasoning: Optional[str] = None  # 仅当解析失败时记录原始输出


# =========================== 主类实现 =========================== #
class RecommendationAgent(BaseAgent):
    """速度优先的推荐 Agent"""

    DEFAULT_TEMPLATE = (
        "You are a recommendation engine for role‑play stories. Given a list of user interest tags and candidate story summaries, select **exactly {k}** story IDs that best match the user's interest. Respond strictly as a JSON list of integers.\n\n"
        "# v1.0\n"
        "### Performance Summary\n"
        "- Current mean precision ≈ 0.492\n"
        "- Current mean recall ≈ 0.492\n"
        "- Δ precision vs prev ≈ 0.050\n"
        "- Δ recall vs prev ≈ 0.050\n\n"
        "### Next Round Focus\n"
        "- Upweight stories with tag: 'k-pop idols' (recently underperformed)\n\n"
        "### Cold Start/Long-tail Issues\n"
        "- The following tags are associated with users who had low precision (<0.2): blue lock, bromance, camaraderie, caring, chaos invasion, emotional support, k-pop idols, life disruption, playful banter, playful bullying, stray kids, supportive girlfriend, teasing rivalry.\n"
        "- Please consider upweighting or diversifying recommendations for these tags to improve cold start/long-tail user experience."
    )

    def __init__(
        self,
        contents_df: pd.DataFrame,
        interactions_df: Optional[pd.DataFrame] = None,
        model_wrapper=None,
        prompt_template: Optional[str] = None,
        k: int = 10,
        max_tokens: int = 8192,
    ):
        """
        Args:
            contents_df: 必含列 {content_id, title, intro}
            interactions_df: 可选列 {user_id, content_id, interaction_count}
            model_wrapper: 若为空，自动 create_recommendation_agent()
            prompt_template: 可被 Prompt‑Optimizer 替换
            k: 默认返回条数
            max_tokens: 模型上下文长度，用于裁剪
        """
        if model_wrapper is None:
            model_wrapper = create_recommendation_agent()

        super().__init__(
            name="RecommendationAgent",
            model_wrapper=model_wrapper,
            config={
                "system_prompt": (
                    "You are a lightning‑fast recommendation assistant. "
                    "Think minimum, output maximum relevance."
                ),
                "temperature": 0.2,
            },
        )

        self.contents_df = contents_df.copy()
        self.interactions_df = (
            interactions_df.groupby("content_id")["interaction_count"].sum()
            if interactions_df is not None and not interactions_df.empty
            else None
        )
        self.k = k
        self.max_tokens = max_tokens
        self.prompt_template = prompt_template or self.DEFAULT_TEMPLATE
        self.prompt_version = "v1.0 (optimized)"

        # 预生成摘要
        self._content_summaries: Dict[int, str] = self._prepare_content_summaries()

        logger.info(
            "RecommendationAgent ready | stories: {} | model: {}",
            len(self._content_summaries),
            model_wrapper.model_name,
        )

    # ------------------------------------------------------------------ #
    #                        核心入口（BaseAgent 标准）                    #
    # ------------------------------------------------------------------ #
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            input_data: {"user_tags": List[str], "num_recommendations": int}
        Returns:
            dict → 可 JSON 序列化
        """
        user_tags: List[str] = [t.lower().strip() for t in input_data["user_tags"]]
        k = int(input_data.get("num_recommendations", self.k))

        # 解析 prompt_template 里的 upweight tag 指令
        upweight_tag = None
        src_text = self.prompt_template
        if "Upweight stories with tag:" in src_text:
            import re
            m = re.search(r"Upweight stories with tag: '([^']+)'", src_text)
            if m:
                upweight_tag = m.group(1).strip().lower()

        # 1) 打分召回候选
        candidates = self._rank_candidates(user_tags, top_n=50, upweight_tag=upweight_tag)

        # 2) 生成 prompt（自动裁剪）
        prompt, prompt_tokens, kept = self._build_prompt(user_tags, candidates, k)

        # 3) 调用 LLM
        llm_output = self.model.generate(
            prompt, system_prompt=self.config["system_prompt"]
        )

        # 4) 解析结果
        content_ids, ok = self._parse_output(llm_output, kept, k)
        if not ok:  # 回退
            content_ids = [c.content_id for c in candidates[:k]]

        result = RecoResult(
            content_ids=content_ids,
            prompt_tokens=prompt_tokens,
            model_tokens=self.model.count_tokens(llm_output),
            model_used=self.model.model_name,
            reasoning=None if ok else llm_output,
        )

        self.log_interaction(
            {"user_tags": user_tags, "k": k},
            asdict(result),
        )
        return asdict(result)

    # ------------------------------------------------------------------ #
    #                           兼容旧测试接口                            #
    # ------------------------------------------------------------------ #
    def recommend(self, request: "RecommendationRequest") -> "RecommendationResponse":
        """
        Wrapper so legacy code can call `agent.recommend(RecommendationRequest)`
        """
        raw = self.process(
            {
                "user_tags": request.user_tags,
                "num_recommendations": request.num_recommendations,
            }
        )
        return RecommendationResponse(
            content_ids=raw["content_ids"],
            reasoning=raw.get("reasoning", ""),
            prompt_version=self.prompt_version,
            model_used=raw["model_used"],
        )

    # ------------------------------------------------------------------ #
    # ----------------------- 核心内部实现（私有） ------------------------ #
    # ------------------------------------------------------------------ #
    def _prepare_content_summaries(self) -> Dict[int, str]:
        summaries = {}
        for _, row in self.contents_df.iterrows():
            cid = int(row["content_id"])
            title = str(row.get("title", "")).strip()
            intro = str(row.get("intro", "")).strip().replace("\n", " ")
            characters = str(row.get("character_list", "")).strip()
            # 拼接更丰富的 story 信息
            summaries[cid] = (
                f"[ID: {cid}] Title: {title}\n"
                f"Intro: {intro[:160]}\n"
                f"Characters: {characters}"
            )
        return summaries

    def _rank_candidates(self, tags: List[str], top_n: int = 50, upweight_tag: Optional[str] = None) -> List[Candidate]:
        scored: List[Candidate] = []
        for cid, summary in self._content_summaries.items():
            tag_score = sum(t in summary.lower() for t in tags)
            if tag_score == 0:
                continue
            # upweight 指定 tag
            if upweight_tag and upweight_tag in summary.lower():
                tag_score += 0.2  # 可调权重
            score = tag_score
            if self.interactions_df is not None and cid in self.interactions_df.index:
                score += np.log(self.interactions_df.loc[cid] + 1)
            scored.append(Candidate(cid, score, summary))
        scored.sort(key=lambda x: (-x.score, x.content_id))
        # Fallback: 若候选不足k，补充热门和多样性内容
        if len(scored) < self.k:
            needed = self.k - len(scored)
            all_ids = set(self._content_summaries.keys())
            existing_ids = set(c.content_id for c in scored)
            missing_ids = list(all_ids - existing_ids)
            if self.interactions_df is not None:
                valid_hot_ids = [cid for cid in missing_ids if cid in self.interactions_df.index]
                hot_ids = list(self.interactions_df.loc[valid_hot_ids].sort_values(ascending=False).index)
            else:
                hot_ids = missing_ids
            diverse_ids = sorted(missing_ids, key=lambda cid: self._content_summaries[cid][10:15])
            fill_ids = hot_ids[:needed//2] + diverse_ids[:needed - needed//2]
            for cid in fill_ids:
                scored.append(Candidate(cid, 0.1, self._content_summaries[cid]))
        # 多样性惩罚：Jaccard(title_i, 已选title) λ=0.2
        lambda_div = 0.2
        def jaccard(a, b):
            set_a = set(a.lower().split())
            set_b = set(b.lower().split())
            return len(set_a & set_b) / (len(set_a | set_b) + 1e-6)
        selected = []
        for cand in scored:
            if len(selected) >= self.k:
                break
            # 计算与已选内容的最大Jaccard
            max_jacc = 0.0
            for sel in selected:
                max_jacc = max(max_jacc, jaccard(cand.summary, sel.summary))
            # 惩罚分
            cand.score -= lambda_div * max_jacc
            selected.append(cand)
        selected.sort(key=lambda x: (-x.score, x.content_id))
        return selected[:top_n]

    def _build_prompt(
        self, tags: List[str], cands: List[Candidate], k: int
    ) -> Tuple[str, int, List[Candidate]]:
        header = "## User Tags\n" + ", ".join(tags) + "\n\n## Candidates\n"

        kept = cands.copy()

        def fmt(cs: List[Candidate]) -> str:
            return "\n".join(c.summary for c in cs)

        prompt_body = self.prompt_template.format(k=k) + "\n\n" + header + fmt(kept)
        tokens = self.model.count_tokens(prompt_body)

        # 留 1 k token 给模型输出
        while tokens > self.max_tokens - 1024 and len(kept) > k:
            kept.pop()
            prompt_body = self.prompt_template.format(k=k) + "\n\n" + header + fmt(kept)
            tokens = self.model.count_tokens(prompt_body)

        return prompt_body, tokens, kept

    @staticmethod
    def _parse_output(
        output: str, kept: List[Candidate], k: int
    ) -> Tuple[List[int], bool]:
        m = re.search(r"\[.*?\]", output, re.DOTALL)
        if not m:
            return [], False
        try:
            candidate_ids = {c.content_id for c in kept}
            ids = [int(x) for x in json.loads(m.group(0))][:k]
            ids = [i for i in ids if i in candidate_ids]
            if len(ids) < k:
                ids.extend(
                    [c.content_id for c in kept if c.content_id not in ids][: k - len(ids)]
                )
            return ids, True
        except Exception:
            return [], False

    # ------------------------------------------------------------------ #
    #                     Prompt 模板热更新（外部可调）                     #
    # ------------------------------------------------------------------ #
    def update_prompt_template(self, new_template: str) -> None:
        self.prompt_template = new_template.strip()
        # 简单递增版本号，便于调试
        major, minor = map(int, self.prompt_version[1:].split("."))
        self.prompt_version = f"v{major}.{minor + 1}"
        # prompt 变更时自动清空模型缓存
        if hasattr(self.model, "clear_cache"):
            self.model.clear_cache()
        logger.success("🔄 Prompt template updated → %s (cache cleared)", self.prompt_version)
