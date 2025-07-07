"""
Recommendation Agent
────────────────────
• 接收用户勾选的兴趣 `tags`，返回 K 个最相关的 `content_id`
• 首要指标：**速度*

功能要点
1. system_prompt / user_prompt 拆分，节省 token
2. token‑budget 自适应裁剪内容摘要，确保 prompt 不超长
3. 关键词命中 + 历史交互次数多信号打分
4. Prompt 模板热更新，供 Optimizer 调用
5. 兼容旧测试：RecommendationRequest / recommend()
6. 向量召回层：FAISS HNSW + all-MiniLM-L6 embedding
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

# 向量召回相关导入
import faiss
from sentence_transformers import SentenceTransformer
import pickle

# ------------------------------------------------------------------ #
#              让 `pytest` 直接运行时也能找到 src 包                 #
# ------------------------------------------------------------------ #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.base import BaseAgent
from src.models.model_wrapper import create_recommendation_agent
from src.agents.multi_view_recall import MultiViewRecall

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
        "You are an expert recommendation assistant for Sekai role-play stories. Your goal is to recommend stories that match the user's interests and preferences.\n\n"
        "## Context\n"
        "Sekai stories are character-driven role-play scenarios, often featuring:\n"
        "- Popular anime/manga IPs (My Hero Academia, One Piece, etc.)\n"
        "- Romantic and relationship dynamics\n"
        "- School and social scenarios\n"
        "- Character interactions and conflicts\n\n"
        "## User Profile\n"
        "User interests: {user_tags}\n\n"
        "## Candidate Stories\n"
        "Below are candidate stories with their titles and descriptions. Select exactly {k} stories that best match the user's interests.\n\n"
        "{candidate_stories}\n\n"
        "## Selection Criteria\n"
        "1. **IP Relevance**: Stories featuring the user's preferred IPs\n"
        "2. **Theme Match**: Stories with themes the user enjoys\n"
        "3. **Character Appeal**: Stories with characters the user likes\n"
        "4. **Diversity**: Balance between familiar and new content\n"
        "5. **Quality**: Well-written and engaging stories\n\n"
        "## Response Format\n"
        "Return ONLY a JSON array of integers, e.g. [123, 456, 789]\n"
        "Do not output any explanation, comments, or text before or after the JSON.\n"
        "Example: [123, 456, 789, ...]\n\n"
        "## Current Performance\n"
        "- Mean precision: 0.492\n"
        "- Mean recall: 0.492\n"
        "- Focus on improving cold-start recommendations\n"
    )

    # 新增：简易重排提示模板
    SIMPLE_RERANK_TEMPLATE = (
        "You are an assistant that re‑orders story IDs for relevance.\n"
        "Input user tags: {tags}\n"
        "Candidate IDs (most relevant first): {ids}\n"
        "Return exactly {k} IDs as a JSON array, no text."
    )

    def __init__(
        self,
        contents_df: pd.DataFrame,
        interactions_df: Optional[pd.DataFrame] = None,
        model_wrapper=None,
        prompt_template: Optional[str] = None,
        k: int = 10,
        max_tokens: int = 8192,
        use_vector_recall: bool = True,
        dry_run: bool = False,
        recall_mode: str = "llm",
        rerank_window: int = 60,  # 增加到60，LLM短prompt+多候选
        use_simple_rerank: bool = True,  # 默认使用简洁模板
        tag_weight: float = 0.1,
        cold_start_boost: float = 0.2,
    ):
        """
        Args:
            contents_df: 必含列 {content_id, title, intro}
            interactions_df: 可选列 {user_id, content_id, interaction_count}
            model_wrapper: 若为空，自动 create_recommendation_agent()
            prompt_template: 可被 Prompt‑Optimizer 替换
            k: 默认返回条数
            max_tokens: 模型上下文长度，用于裁剪
            use_vector_recall: 是否启用向量召回
            dry_run: 是否为 dry_run 模式
            recall_mode: 'vector' 只用向量, 'llm' 向量+LLM重排
            rerank_window: LLM重排候选数
            use_simple_rerank: 是否使用简易重排提示模板
            tag_weight: 用于标签重排的权重
            cold_start_boost: 用于冷启动标签的权重
        """
        if model_wrapper is None and recall_mode != "vector":
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
        self.use_vector_recall = use_vector_recall
        self.dry_run = dry_run
        self.recall_mode = recall_mode
        self.rerank_window = rerank_window
        self.use_simple_rerank = use_simple_rerank
        self.tag_weight = tag_weight
        self.cold_start_boost = cold_start_boost
        # Example: hardcoded cold start tags, can be made configurable
        self.cold_start_tags = {"blue lock", "k-pop idols", "stray kids"}
        # 选择模板
        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = self.SIMPLE_RERANK_TEMPLATE  # 默认使用简洁模板
        self.prompt_version = "v1.0 (optimized)"

        # 预生成摘要
        self._content_summaries: Dict[int, str] = self._prepare_content_summaries()

        # 向量召回相关初始化
        self._faiss_index = None
        self._embedding_model = None
        self._content_embeddings = None
        self._content_ids_list = None
        
        if self.use_vector_recall:
            self._build_faiss_index()

        # 新增：为每个内容提取标签
        self._story_tags = self._extract_story_tags()
        
        # 新增：多视角召回系统
        self.multi_view_recall = MultiViewRecall(contents_df, interactions_df)

        print(f"[RecommendationAgent] tag_weight={self.tag_weight}, cold_start_boost={self.cold_start_boost}, recall_mode={self.recall_mode}, rerank_window={self.rerank_window}")
        print(f"[DEBUG] contents_df shape: {contents_df.shape}")
        print(f"[DEBUG] Content IDs range: {contents_df['content_id'].min()} - {contents_df['content_id'].max()}")
        print(f"[DEBUG] Sample content IDs: {contents_df['content_id'].head(10).tolist()}")

        logger.info(
            "RecommendationAgent ready | stories: {} | model: {} | vector_recall: {}",
            len(self._content_summaries),
            model_wrapper.model_name if model_wrapper else "vector-only",
            self.use_vector_recall,
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
        print(f"[RecommendationAgent.process] recall_mode={self.recall_mode}")
        assert self.recall_mode in ("vector", "llm"), f"recall_mode异常: {self.recall_mode}"
        user_tags: List[str] = [t.lower().strip() for t in input_data["user_tags"]]
        k = int(input_data.get("num_recommendations", self.k))
        upweight_tag = None
        src_text = self.prompt_template
        if "Upweight stories with tag:" in src_text:
            import re
            m = re.search(r"Upweight stories with tag: '([^']+)'", src_text)
            if m:
                upweight_tag = m.group(1).strip().lower()
        # --- recall_mode 分流 ---
        if self.recall_mode == "vector":
            print("[process] Using VECTOR recall branch")
            vector_candidates = self._vector_recall(user_tags, top_k=k)
            content_ids = [cid for cid, _ in vector_candidates][:k]
            result = RecoResult(
                content_ids=content_ids,
                prompt_tokens=0,
                model_tokens=0,
                model_used="vector-only",
                reasoning="recall_mode=vector"
            )
            result_dict = asdict(result)
            result_dict["method_used"] = "vector_recall"
            self.log_interaction({"user_tags": user_tags, "k": k}, result_dict)
            return result_dict
        elif self.recall_mode == "llm":
            print("[process] Using LLM rerank branch")
            vector_candidates = self._vector_recall(user_tags, top_k=120)
            scored = []
            user_tag_set = set(user_tags)
            for cid, score in vector_candidates:
                story_tags = self._story_tags.get(cid, set())
                overlap = len(user_tag_set & story_tags) / max(1, len(user_tag_set))
                boost = 1.0 + overlap * self.tag_weight
                if len(story_tags & self.cold_start_tags) > 0:
                    boost *= (1.0 + self.cold_start_boost)
                scored.append((cid, score * boost))
            scored = sorted(scored, key=lambda x: -x[1])[:self.rerank_window]
            candidates = []
            for cid, score in scored:
                summary = self._content_summaries.get(cid, "")
                candidates.append(Candidate(cid, score, summary))
            prompt, prompt_tokens, kept = self._build_prompt(user_tags, candidates, k)
            logger.info(f"[LLM Rerank] Prompt length: {prompt_tokens} tokens | rerank_window={self.rerank_window}")
            print(f"[DEBUG] Candidates passed to LLM (first 5): {[c.content_id for c in candidates[:5]]}")
            llm_output = self.model.generate(
                prompt, system_prompt=self.config["system_prompt"]
            )
            content_ids, ok = self._parse_output(llm_output, kept, k)
            content_ids = list(dict.fromkeys(content_ids))
            if len(content_ids) < k:
                all_ids = set(c.content_id for c in candidates)
                needed = k - len(content_ids)
                extra_ids = [cid for cid in self._content_summaries if cid not in all_ids][:needed]
                content_ids.extend(extra_ids)
            content_ids = content_ids[:k]
            if not ok:
                print(f"[DEBUG] LLM output parse failed, raw output: {llm_output}")
                content_ids = [c.content_id for c in candidates[:k]]
            result = RecoResult(
                content_ids=content_ids,
                prompt_tokens=prompt_tokens,
                model_tokens=self.model.count_tokens(llm_output),
                model_used=self.model.model_name,
                reasoning=None if ok else llm_output,
            )
            result_dict = asdict(result)
            result_dict["method_used"] = "llm_rerank"
            self.log_interaction({"user_tags": user_tags, "k": k}, result_dict)
            return result_dict
        print("[process] Using FALLBACK branch")
        candidates = self._recall_candidates(user_tags, top_n=50, upweight_tag=upweight_tag)
        prompt, prompt_tokens, kept = self._build_prompt(user_tags, candidates, k)
        llm_output = self.model.generate(
            prompt, system_prompt=self.config["system_prompt"]
        )
        content_ids, ok = self._parse_output(llm_output, kept, k)
        content_ids = list(dict.fromkeys(content_ids))
        if len(content_ids) < k:
            all_ids = set(c.content_id for c in candidates)
            needed = k - len(content_ids)
            extra_ids = [cid for cid in self._content_summaries if cid not in all_ids][:needed]
            content_ids.extend(extra_ids)
        content_ids = content_ids[:k]
        if not ok:
            content_ids = [c.content_id for c in candidates[:k]]
        result = RecoResult(
            content_ids=content_ids,
            prompt_tokens=prompt_tokens,
            model_tokens=self.model.count_tokens(llm_output),
            model_used=self.model.model_name,
            reasoning=None if ok else llm_output,
        )
        result_dict = asdict(result)
        result_dict["method_used"] = "fallback"
        self.log_interaction({"user_tags": user_tags, "k": k}, result_dict)
        return result_dict

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

    def _extract_story_tags(self):
        # 假设内容摘要中有标签字段，否则可自定义提取逻辑
        tags_dict = {}
        for cid, summary in self._content_summaries.items():
            # 简单正则提取标签，或根据实际数据结构调整
            # 这里假设标签以 'Tags: xxx, yyy' 出现
            import re
            m = re.search(r"Tags?: ([^\n]+)", summary)
            if m:
                tags = [t.strip().lower() for t in m.group(1).split(",") if t.strip()]
            else:
                tags = []
            tags_dict[cid] = set(tags)
        return tags_dict

    def _rank_candidates(self, tags: List[str], top_n: int = 50, upweight_tag: Optional[str] = None) -> List[Candidate]:
        scored: List[Candidate] = []
        # --- 新增：去重内容ID ---
        seen = set()
        unique_candidates = []
        for cid, summary in self._content_summaries.items():
            if cid in seen:
                continue
            seen.add(cid)
            tag_score = sum(t in summary.lower() for t in tags)
            if tag_score == 0:
                continue
            # upweight 指定 tag
            if upweight_tag and upweight_tag in summary.lower():
                tag_score += 0.2  # 可调权重
            score = tag_score
            if self.interactions_df is not None and cid in self.interactions_df.index:
                score += np.log(self.interactions_df.loc[cid] + 1)
            unique_candidates.append(Candidate(cid, score, summary))
        scored = unique_candidates
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

        # 修复：根据模板类型正确填充 prompt
        if "tags" in self.prompt_template and "ids" in self.prompt_template:
            # 简洁模板格式
            prompt_body = self.prompt_template.format(
                tags=", ".join(tags),
                ids=", ".join(str(c.content_id) for c in kept),
                k=k
            )
        else:
            # 完整模板格式
            prompt_body = self.prompt_template.format(
                user_tags=", ".join(tags),
                candidate_stories=header + fmt(kept),
                k=k
            )
        tokens = self.model.count_tokens(prompt_body)
        # 留 1 k token 给模型输出
        while tokens > self.max_tokens - 1024 and len(kept) > k:
            kept.pop()
            if "tags" in self.prompt_template and "ids" in self.prompt_template:
                # 简洁模板格式
                prompt_body = self.prompt_template.format(
                    tags=", ".join(tags),
                    ids=", ".join(str(c.content_id) for c in kept),
                    k=k
                )
            else:
                # 完整模板格式
                prompt_body = self.prompt_template.format(
                    user_tags=", ".join(tags),
                    candidate_stories=header + fmt(kept),
                    k=k
                )
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
        # 处理版本号格式 "v1.0 (optimized)" -> 提取 "1.0"
        version_part = self.prompt_version[1:].split(" ")[0]  # 去掉 "v" 和 " (optimized)"
        major, minor = map(int, version_part.split("."))
        self.prompt_version = f"v{major}.{minor + 1} (optimized)"
        # prompt 变更时自动清空模型缓存
        if hasattr(self.model, "clear_cache"):
            self.model.clear_cache()
        logger.success("🔄 Prompt template updated → %s (cache cleared)", self.prompt_version)

    # ------------------------------------------------------------------ #
    # ----------------------- 向量召回层实现 ----------------------------- #
    # ------------------------------------------------------------------ #
    def _build_faiss_index(self) -> None:
        """构建FAISS HNSW索引用于向量召回"""
        try:
            logger.info("🔧 Building FAISS index for vector recall...")
            
            # 初始化embedding模型
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # 准备文本数据
            content_texts = []
            self._content_ids_list = []
            
            for cid, summary in self._content_summaries.items():
                # 提取标题和简介作为embedding文本
                title = self.contents_df.loc[self.contents_df['content_id'] == cid, 'title'].iloc[0] if 'title' in self.contents_df.columns else ""
                intro = self.contents_df.loc[self.contents_df['content_id'] == cid, 'intro'].iloc[0] if 'intro' in self.contents_df.columns else ""
                text = f"{title} {intro}".strip()
                if text:
                    content_texts.append(text)
                    self._content_ids_list.append(cid)
            
            if not content_texts:
                logger.warning("⚠️ No valid content texts found for embedding")
                return
            
            # 生成embeddings
            logger.info(f"📊 Generating embeddings for {len(content_texts)} contents...")
            self._content_embeddings = self._embedding_model.encode(content_texts, show_progress_bar=True)
            
            # 构建FAISS HNSW索引
            dimension = self._content_embeddings.shape[1]
            self._faiss_index = faiss.IndexHNSWFlat(dimension, 32)  # 32个邻居
            self._faiss_index.hnsw.efConstruction = 200  # 构建时的搜索深度
            self._faiss_index.hnsw.efSearch = 100  # 搜索时的深度
            
            # 添加向量到索引
            self._faiss_index.add(self._content_embeddings.astype('float32'))
            
            print(f"[DEBUG] Content IDs in FAISS: {self._content_ids_list[:5]} ... {self._content_ids_list[-5:]}")
            
            logger.success(f"✅ FAISS index built successfully | dimension: {dimension} | vectors: {len(self._content_ids_list)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to build FAISS index: {e}")
            self.use_vector_recall = False

    def _vector_recall(self, user_tags: List[str], top_k: int = 500) -> List[Tuple[int, float]]:
        """多视角召回（替换原来的向量召回）"""
        print(f"[DEBUG] User tags: {user_tags}")
        try:
            results = self.multi_view_recall.recall(user_tags, top_k)
            vector_results = [(content_id, score) for content_id, score, method in results]
            print(f"[DEBUG] Retrieved candidates (first 5): {vector_results[:5]}")
            return vector_results
        except Exception as e:
            print(f"[DEBUG] Multi-view recall failed: {e}")
            # 回退到原来的向量召回
            if self.use_vector_recall and self._faiss_index is not None and self._embedding_model is not None and self._content_ids_list is not None:
                try:
                    # 将用户标签组合成查询文本
                    query_text = " ".join(user_tags)
                    
                    # 生成查询embedding
                    query_embedding = self._embedding_model.encode([query_text])
                    
                    # FAISS搜索
                    scores, indices = self._faiss_index.search(query_embedding.astype('float32'), top_k)
                    
                    # 返回(content_id, score)对
                    results = []
                    for idx, score in zip(indices[0], scores[0]):
                        if idx < len(self._content_ids_list):
                            content_id = self._content_ids_list[idx]
                            results.append((content_id, float(score)))
                    
                    logger.debug(f"🔍 Fallback vector recall returned {len(results)} candidates")
                    return results
                except Exception as e2:
                    logger.error(f"Fallback vector recall also failed: {e2}")
            
            return []

    def _recall_candidates(self, user_tags: List[str], top_n: int = 50, upweight_tag: Optional[str] = None, fusion_weights: Optional[Tuple[float, float]] = None) -> List[Candidate]:
        """融合关键词和向量召回的候选生成，支持动态融合权重"""
        candidates_dict = {}
        # 1. 关键词召回
        keyword_candidates = self._rank_candidates(user_tags, top_n=top_n, upweight_tag=upweight_tag)
        for cand in keyword_candidates:
            candidates_dict[cand.content_id] = {
                'keyword_score': cand.score,
                'vector_score': 0.0,
                'summary': cand.summary
            }
        # 2. 向量召回
        if self.use_vector_recall:
            vector_results = self._vector_recall(user_tags, top_k=500)
            for content_id, vector_score in vector_results:
                if content_id in candidates_dict:
                    candidates_dict[content_id]['vector_score'] = vector_score
                else:
                    summary = self._content_summaries.get(content_id, "")
                    candidates_dict[content_id] = {
                        'keyword_score': 0.0,
                        'vector_score': vector_score,
                        'summary': summary
                    }
        # 3. 融合分数：支持动态权重
        if fusion_weights is None:
            fusion_weights = getattr(self, "fusion_weights", (0.3, 0.7))  # 向量权重 ≥0.5
        hotness_w, vector_w = fusion_weights
        final_candidates = []
        for content_id, scores in candidates_dict.items():
            if scores['summary']:
                normalized_vector_score = min(max(scores['vector_score'], 0), 1)
                final_score = hotness_w * scores['keyword_score'] + vector_w * normalized_vector_score
                final_candidates.append(Candidate(
                    content_id=content_id,
                    score=final_score,
                    summary=scores['summary']
                ))
        final_candidates.sort(key=lambda x: (-x.score, x.content_id))
        logger.info(f"🎯 Recall candidates: {len(final_candidates)} total | vector_recall={self.use_vector_recall} | fusion_weights={fusion_weights}")
        return final_candidates[:top_n]

    def grid_search_fusion_weights(self, eval_agent, users_df, k=10, weights_grid=[(0.3,0.7),(0.5,0.5),(0.7,0.3)]):
        """对融合权重进行网格搜索，返回最佳权重和对应P@10"""
        best_p10 = -1
        best_weights = None
        results = []
        for hotness_w, vector_w in weights_grid:
            p10s = []
            for _, urow in users_df.iterrows():
                user_dict = urow.to_dict()
                user_tags = [t.lower().strip() for t in user_dict.get("tags", [])]
                cands = self._recall_candidates(user_tags, top_n=k, fusion_weights=(hotness_w, vector_w))
                rec_ids = [c.content_id for c in cands]
                gt_ids = user_dict.get("ground_truth", [])
                if not gt_ids:
                    continue
                hit = len(set(rec_ids) & set(gt_ids))
                p10s.append(hit / k)
            mean_p10 = sum(p10s) / len(p10s) if p10s else 0
            results.append({"weights": (hotness_w, vector_w), "p10": mean_p10})
            if mean_p10 > best_p10:
                best_p10 = mean_p10
                best_weights = (hotness_w, vector_w)
        logger.info(f"[GridSearch] Fusion weights results: {results}")
        logger.success(f"[GridSearch] Best fusion weights: {best_weights} | P@10={best_p10:.3f}")
        self.fusion_weights = best_weights if best_weights is not None else (0.3, 0.7)  # 向量权重 ≥0.5
        return best_weights, best_p10, results
