"""
Evaluation Agent
================
1. 读取 **完整用户画像**（users.csv 中的一行 / dict）
2. **模拟**用户首屏勾选的 tags
3. 调用传入的 RecommendationAgent 获取 top‑k 推荐
4. 用同一批 tags + 全量 story pool 计算 "ground‑truth" 列表
5. 输出 precision@k 及相关明细

依赖：
    - BaseAgent     (src/agents/base.py)
    - create_evaluation_agent (src/models/model_wrapper.py)
    - RecommendationAgent    (由调用方注入，方便测试 & 解耦外部 API)
"""

from __future__ import annotations

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
from loguru import logger #type: ignore
import numpy as np

# 如有可用则导入sentence-transformers，否则提示安装
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    logger.warning("sentence-transformers 未安装，embedding通道不可用")

# 将项目根目录加入 sys.path，便于 tests 直接运行
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.base import BaseAgent
from src.models.model_wrapper import create_evaluation_agent


@dataclass
class EvalResult:
    """Evaluation 输出结构，方便类型提示与序列化"""
    user_id: int
    simulated_tags: List[str]
    recommended: List[int]
    ground_truth: List[int]
    precision_at_k: float
    recall_at_k: float
    model_used: str
    method_used: str  # "llm" or "keyword"


# 评估指标注册表和注册装饰器
METRIC_REGISTRY = {}

def register_metric(name):
    def decorator(fn):
        METRIC_REGISTRY[name] = fn
        return fn
    return decorator

@register_metric("precision")
def precision_at_k(recommended, ground_truth, k):
    if not recommended:
        return 0.0
    hit = len(set(recommended[:k]) & set(ground_truth))
    return hit / k

@register_metric("recall")
def recall_at_k(recommended, ground_truth, k):
    if not ground_truth:
        return 0.0
    hit = len(set(recommended[:k]) & set(ground_truth))
    return hit / len(ground_truth)

@register_metric("f1")
def f1_at_k(recommended, ground_truth, k):
    p = precision_at_k(recommended, ground_truth, k)
    r = recall_at_k(recommended, ground_truth, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


class EvaluationAgent(BaseAgent):
    """负责评估 RecommendationAgent 输出质量"""

    def __init__(
        self,
        users_df: pd.DataFrame,
        contents_df: pd.DataFrame,
        recommendation_agent: BaseAgent,
        model_wrapper=None,
        k: int = 10,
        use_llm_for_ground_truth: bool = False,
        use_llm_for_tag_simulation: bool = True,
        eval_mode: str = "llm",
    ):
        if model_wrapper is None:
            model_wrapper = create_evaluation_agent()

        super().__init__(
            name="EvaluationAgent",
            model_wrapper=model_wrapper,
            config={
                "system_prompt": self._system_prompt(),
                "temperature": 0.3,
            },
        )

        self.users_df = users_df
        self.contents_df = contents_df
        self.reco_agent = recommendation_agent
        self.k = k
        self.use_llm_for_ground_truth = use_llm_for_ground_truth
        self.use_llm_for_tag_simulation = use_llm_for_tag_simulation
        self.eval_mode = eval_mode

        # 预处理：提取所有可能的标签
        self._all_user_tags = self._extract_all_tags()

        self._embedding_model = None
        self._content_embeddings = None
        self._content_id_list = None
        self._build_content_embeddings()

        logger.info(
            f"EvaluationAgent 就绪 | 用户数: {len(users_df)} | 内容数: {len(contents_df)} | "
            f"评估模型: {model_wrapper.model_name} | LLM GT: {use_llm_for_ground_truth} | "
            f"LLM Tags: {use_llm_for_tag_simulation}"
        )

    # --------------------------------------------------------------------- #
    # --------------------------- 核心流程方法 ----------------------------- #
    # --------------------------------------------------------------------- #

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        BaseAgent 统一入口：`input_data` 需包含 `user_profile` (dict)
        """
        user_profile: Dict[str, Any] = input_data["user_profile"]
        result = self.evaluate(user_profile)

        # 写交互日志
        self.log_interaction(user_profile, asdict(result))
        return asdict(result)

    # ---------------------------- 标签处理 ---------------------------------- #
    
    def _extract_all_tags(self) -> set:
        """从所有用户数据中提取所有可能的标签"""
        all_tags = set()
        for _, user in self.users_df.iterrows():
            tags = [t.strip().lower() for t in str(user.get('user_interest_tags', '')).split(',') if t.strip()]
            all_tags.update(tags)
        return all_tags

    def simulate_tags(self, user_profile: Dict[str, Any]) -> List[str]:
        # 省费模式：直接读取 sim_tags
        if 'sim_tags' in user_profile and user_profile['sim_tags']:
            if isinstance(user_profile['sim_tags'], str):
                tags = [t.strip().lower() for t in user_profile['sim_tags'].split(',') if t.strip()]
            else:
                tags = [t.strip().lower() for t in user_profile['sim_tags'] if t.strip()]
            return tags[:7]
        # 读取原始用户兴趣标签
        elif 'user_interest_tags' in user_profile and user_profile['user_interest_tags']:
            if isinstance(user_profile['user_interest_tags'], str):
                tags = [t.strip().lower() for t in user_profile['user_interest_tags'].split(',') if t.strip()]
            else:
                tags = [t.strip().lower() for t in user_profile['user_interest_tags'] if t.strip()]
            return tags[:7]
        # fallback: 兼容老逻辑
        return []

    # ---------------------------- Ground Truth 生成 ---------------------------------- #

    def _build_content_embeddings(self):
        if SentenceTransformer is None:
            logger.warning("sentence-transformers 未安装，embedding通道不可用")
            return
        logger.info("[Eval] 构建内容embedding...")
        content_texts = []
        content_ids = []
        for _, row in self.contents_df.iterrows():
            cid = int(row["content_id"])
            title = str(row.get("title", "")).strip()
            intro = str(row.get("intro", "")).strip()
            text = f"{title} {intro}".strip()
            if text:
                content_texts.append(text)
                content_ids.append(cid)
        if not content_texts:
            logger.warning("[Eval] 无有效内容文本，跳过embedding构建")
            return
        self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self._content_embeddings = self._embedding_model.encode(content_texts, show_progress_bar=False)
        self._content_id_list = content_ids
        logger.info(f"[Eval] embedding构建完成，内容数: {len(content_ids)}")

    def _embedding_recall(self, user_tags: List[str], top_n: int = 50) -> List[int]:
        if self._embedding_model is None or self._content_embeddings is None:
            return []
        query_text = " ".join(user_tags)
        query_emb = self._embedding_model.encode([query_text])[0]
        # 计算cosine相似度
        scores = np.dot(self._content_embeddings, query_emb) / (
            np.linalg.norm(self._content_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8)
        top_idx = np.argsort(scores)[::-1][:top_n]
        return [self._content_id_list[i] for i in top_idx]

    def make_ground_truth(self, user_profile: Dict[str, Any], top_n: int) -> List[int]:
        # 省费模式：直接用embedding余弦排序
        tags = self.simulate_tags(user_profile)
        candidate_ids = self._embedding_recall(tags, top_n=150)
        return candidate_ids[:top_n]

    # ---------------------------- 主评估流程 ---------------------------------- #

    def evaluate(self, user_profile: Dict[str, Any]) -> EvalResult:
        """
        综合调用以上步骤完成一次评估
        """
        # 1. 模拟标签
        tags = self.simulate_tags(user_profile)
        logger.debug(f"Simulated tags for user {user_profile.get('user_id')}: {tags}")

        # 2. 获取推荐结果
        reco_output = self.reco_agent.process(
            {"user_tags": tags, "num_recommendations": self.k}
        )
        recommended_ids = reco_output.get("content_ids", [])[: self.k]
        logger.debug(f"Recommended IDs: {recommended_ids}")

        # 3. 生成 ground truth
        ground_truth = self.make_ground_truth(user_profile, self.k)
        logger.debug(f"Ground truth IDs: {ground_truth}")

        # 4. 计算指标
        k = self.k
        metrics = {}
        for metric_name in ["precision", "recall", "f1"]:
            metric_fn = METRIC_REGISTRY[metric_name]
            metrics[f"{metric_name}_at_k"] = metric_fn(recommended_ids, ground_truth, k)

        # 5. 确定使用的方法
        method_used = reco_output.get("method_used", "unknown")

        print("∩(recommended, GT) =", len(set(recommended_ids) & set(ground_truth)))
        print("recommended =", recommended_ids)
        print("ground_truth =", ground_truth)

        return EvalResult(
            user_id=int(user_profile.get("user_id", -1)),
            simulated_tags=tags,
            recommended=recommended_ids,
            ground_truth=ground_truth,
            precision_at_k=metrics["precision_at_k"],
            recall_at_k=metrics["recall_at_k"],
            model_used=self.model.model_name,
            method_used=method_used
        )

    # --------------------------------------------------------------------- #
    # -------------------------- Prompt Section --------------------------- #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _system_prompt() -> str:
        """供高级 LLM 判断 ground‑truth 时使用"""
        return (
            "You are a senior evaluation agent for a role‑play story platform. "
            "You will analyze user preferences and story content to make precise matches. "
            "Always focus on the user's explicit interests and implied preferences. "
            "Be objective and consistent in your evaluations."
        )