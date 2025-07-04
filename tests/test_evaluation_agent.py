import sys
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import pytest

# 让测试脚本可直接找到 src 包
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.base import BaseAgent
from src.agents.evaluation_agent import EvaluationAgent, EvalResult


# ------------------------------------------------------------------ #
# ----------------------- Dummy / Stub Classes --------------------- #
# ------------------------------------------------------------------ #
class DummyModelWrapper:
    """最小化实现，满足 BaseAgent 依赖"""

    def __init__(self, name: str = "dummy-model"):
        self.model_name = name

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        # 模拟 LLM 响应
        if "select 3-7 tags" in prompt:
            # 模拟标签生成
            return "romance, ninja, adventure"
        elif "select EXACTLY" in prompt and "story IDs" in prompt:
            # 模拟 ground truth 生成
            return "[1, 3, 2]"
        return ""

    def count_tokens(self, text: str) -> int:
        return len(text) // 4


class DummyRecommendationAgent(BaseAgent):
    """固定返回预设推荐 ID 列表"""

    def __init__(self, fixed_ids: List[int]):
        super().__init__("DummyRecoAgent", DummyModelWrapper(), {})
        self._ids = fixed_ids

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        k = input_data.get("num_recommendations", 10)
        return {"content_ids": self._ids[:k]}


# ------------------------------------------------------------------ #
# --------------------------- Fixtures ----------------------------- #
# ------------------------------------------------------------------ #
@pytest.fixture(scope="module")
def sample_data():
    # contents
    contents_df = pd.DataFrame(
        [
            {
                "content_id": 1,
                "title": "Romantic Ninja",
                "intro": "A heart‑warming romance ninja story",
                "character_list": "Naruto, Hinata"
            },
            {
                "content_id": 2,
                "title": "Space Adventure",
                "intro": "High‑octane sci‑fi action among the stars",
                "character_list": "Captain Kirk, Spock"
            },
            {
                "content_id": 3,
                "title": "Slice of Life",
                "intro": "Everyday wholesome moments",
                "character_list": "Original Characters"
            },
        ]
    )

    # users
    users_df = pd.DataFrame(
        [
            {
                "user_id": 100,
                "user_interest_tags": "romance, ninja, adventure, naruto",
            },
            {
                "user_id": 101,
                "user_interest_tags": "sci-fi, space, action, star trek",
            }
        ]
    )

    return users_df, contents_df


@pytest.fixture
def eval_agent(sample_data):
    users_df, contents_df = sample_data
    dummy_reco = DummyRecommendationAgent([1, 2, 3])
    agent = EvaluationAgent(
        users_df=users_df,
        contents_df=contents_df,
        recommendation_agent=dummy_reco,
        model_wrapper=DummyModelWrapper(),
        k=3,
        use_llm_for_ground_truth=False,  # 使用简单方法以便测试
        use_llm_for_tag_simulation=False
    )
    return agent, users_df


@pytest.fixture
def eval_agent_with_llm(sample_data):
    users_df, contents_df = sample_data
    dummy_reco = DummyRecommendationAgent([1, 2, 3])
    agent = EvaluationAgent(
        users_df=users_df,
        contents_df=contents_df,
        recommendation_agent=dummy_reco,
        model_wrapper=DummyModelWrapper(),
        k=3,
        use_llm_for_ground_truth=True,
        use_llm_for_tag_simulation=True
    )
    return agent, users_df


# ------------------------------------------------------------------ #
# --------------------------- Unit Tests --------------------------- #
# ------------------------------------------------------------------ #

def test_simulate_tags_simple(eval_agent):
    """测试简单标签模拟"""
    agent, users_df = eval_agent
    user_profile = users_df.iloc[0].to_dict()
    tags = agent._simulate_tags_simple(user_profile)
    assert set(tags) == {"romance", "ninja", "adventure", "naruto"}


def test_simulate_tags_with_llm(eval_agent_with_llm):
    """测试 LLM 标签模拟"""
    agent, users_df = eval_agent_with_llm
    user_profile = users_df.iloc[0].to_dict()
    tags = agent._simulate_tags_with_llm(user_profile)
    assert isinstance(tags, list)
    assert len(tags) >= 3
    assert len(tags) <= 7


def test_score_metrics():
    """测试评分函数"""
    from src.agents.evaluation_agent import EvaluationAgent as _EA

    # 测试 precision 和 recall
    precision, recall = _EA.score([1, 2, 3], [2, 3, 4])
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    
    # 测试空列表
    precision, recall = _EA.score([], [1])
    assert precision == 0.0
    assert recall == 0.0
    
    # 测试完全匹配
    precision, recall = _EA.score([1, 2, 3], [1, 2, 3])
    assert precision == 1.0
    assert recall == 1.0


def test_make_ground_truth_keywords(eval_agent):
    """测试关键词方法生成 ground truth"""
    agent, users_df = eval_agent
    user_profile = users_df.iloc[0].to_dict()
    ground_truth = agent._make_ground_truth_with_keywords(user_profile, 3)
    
    assert isinstance(ground_truth, list)
    assert len(ground_truth) == 3
    # 应该包含与 ninja 和 romance 相关的内容
    assert 1 in ground_truth  # Romantic Ninja


def test_make_ground_truth_llm(eval_agent_with_llm):
    """测试 LLM 方法生成 ground truth"""
    agent, users_df = eval_agent_with_llm
    user_profile = users_df.iloc[0].to_dict()
    ground_truth = agent._make_ground_truth_with_llm(user_profile, 3)
    
    assert isinstance(ground_truth, list)
    assert len(ground_truth) == 3
    # 根据 dummy 模型的返回，应该是 [1, 3, 2]
    assert ground_truth == [1, 3, 2]


def test_full_evaluate_flow(eval_agent):
    """测试完整评估流程"""
    agent, users_df = eval_agent
    result = agent.evaluate(users_df.iloc[0].to_dict())

    # 断言输出字段完整（注意新增了 recall_at_k 和 method_used）
    expected_fields = [
        "user_id",
        "simulated_tags",
        "recommended",
        "ground_truth",
        "precision_at_k",
        "recall_at_k",
        "model_used",
        "method_used"
    ]
    
    for field in expected_fields:
        assert hasattr(result, field), f"Missing field: {field}"

    # 值的合理性检查
    assert result.user_id == 100
    assert 0.0 <= result.precision_at_k <= 1.0
    assert 0.0 <= result.recall_at_k <= 1.0
    assert result.method_used in ["llm", "keyword"]
    assert isinstance(result.simulated_tags, list)
    assert isinstance(result.recommended, list)
    assert isinstance(result.ground_truth, list)


def test_evaluate_with_different_users(eval_agent):
    """测试不同用户的评估"""
    agent, users_df = eval_agent
    
    # 测试第一个用户（偏好 ninja/romance）
    result1 = agent.evaluate(users_df.iloc[0].to_dict())
    
    # 测试第二个用户（偏好 sci-fi/space）
    result2 = agent.evaluate(users_df.iloc[1].to_dict())
    
    # 两个用户应该有不同的标签
    assert result1.simulated_tags != result2.simulated_tags
    
    # ground truth 也应该不同
    assert result1.ground_truth != result2.ground_truth


def test_process_method(eval_agent):
    """测试 BaseAgent 的 process 方法"""
    agent, users_df = eval_agent
    
    input_data = {"user_profile": users_df.iloc[0].to_dict()}
    output = agent.process(input_data)
    
    # 应该返回字典格式
    assert isinstance(output, dict)
    assert "user_id" in output
    assert "precision_at_k" in output
    assert "recall_at_k" in output


def test_extract_all_tags(eval_agent):
    """测试标签提取功能"""
    agent, _ = eval_agent
    all_tags = agent._all_user_tags
    
    assert isinstance(all_tags, set)
    # 应该包含示例数据中的所有标签
    expected_tags = {"romance", "ninja", "adventure", "naruto", "sci-fi", "space", "action", "star trek"}
    assert expected_tags.issubset(all_tags)


def test_filter_content_by_tags(eval_agent):
    """测试基于标签的内容过滤"""
    agent, _ = eval_agent
    
    # 测试 ninja 相关标签
    filtered = agent._filter_content_by_tags(["ninja", "romance"])
    assert 1 in filtered  # Romantic Ninja 应该排第一
    
    # 测试 space 相关标签
    filtered = agent._filter_content_by_tags(["space", "sci-fi"])
    assert 2 in filtered  # Space Adventure 应该被选中


# ------------------------------------------------------------------ #
# ----------------------- Integration Tests ------------------------ #
# ------------------------------------------------------------------ #

def test_integration_with_real_recommendation(sample_data):
    """集成测试：使用真实的推荐结果"""
    users_df, contents_df = sample_data
    
    # 创建一个会根据标签返回不同结果的推荐 agent
    class SmartDummyRecommendationAgent(BaseAgent):
        def __init__(self):
            super().__init__("SmartDummyRecoAgent", DummyModelWrapper(), {})
        
        def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            tags = input_data.get("user_tags", [])
            k = input_data.get("num_recommendations", 10)
            
            # 根据标签返回不同的推荐
            if "ninja" in tags or "romance" in tags:
                return {"content_ids": [1, 3, 2][:k]}
            elif "space" in tags or "sci-fi" in tags:
                return {"content_ids": [2, 3, 1][:k]}
            else:
                return {"content_ids": [3, 1, 2][:k]}
    
    smart_reco = SmartDummyRecommendationAgent()
    eval_agent = EvaluationAgent(
        users_df=users_df,
        contents_df=contents_df,
        recommendation_agent=smart_reco,
        model_wrapper=DummyModelWrapper(),
        k=3,
        use_llm_for_ground_truth=False,
        use_llm_for_tag_simulation=False
    )
    
    # 评估第一个用户
    result = eval_agent.evaluate(users_df.iloc[0].to_dict())
    
    # 推荐应该偏向 ninja/romance 内容
    assert 1 in result.recommended  # Romantic Ninja
    assert result.precision_at_k > 0  # 应该有一些匹配


# ------------------------------------------------------------------ #
# ------------------------- Parametrized Tests --------------------- #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("use_llm_gt,use_llm_tags", [
    (False, False),
    (True, False),
    (False, True),
    (True, True),
])
def test_different_llm_configurations(sample_data, use_llm_gt, use_llm_tags):
    """测试不同的 LLM 配置组合"""
    users_df, contents_df = sample_data
    dummy_reco = DummyRecommendationAgent([1, 2, 3])
    
    agent = EvaluationAgent(
        users_df=users_df,
        contents_df=contents_df,
        recommendation_agent=dummy_reco,
        model_wrapper=DummyModelWrapper(),
        k=3,
        use_llm_for_ground_truth=use_llm_gt,
        use_llm_for_tag_simulation=use_llm_tags
    )
    
    result = agent.evaluate(users_df.iloc[0].to_dict())
    
    # 无论哪种配置，都应该产生有效结果
    assert isinstance(result, EvalResult)
    assert len(result.recommended) == 3
    assert len(result.ground_truth) == 3
    assert result.method_used == ("llm" if use_llm_gt else "keyword")