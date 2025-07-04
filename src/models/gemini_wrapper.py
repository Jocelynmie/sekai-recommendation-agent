"""
src/models/gemini_wrapper.py
────────────────────────────
• 纯本地 Stub，用于在无 API‑Key 环境下跑测试
• 公开接口与旧版测试脚本保持一致
"""

from typing import Optional, Dict, Any


class GeminiWrapper:
    """足够让测试通过的最小实现"""

    MODELS = {
        "flash": "gemini-2.0-flash-exp",
        "pro": "gemini-1.5-pro-002",
        "flash-thinking": "gemini-2.0-flash-thinking-exp",
    }

    def __init__(self, model_type: str = "flash", temperature: float = 0.3):
        self.model_type = model_type
        self.model_name = self.MODELS.get(model_type, f"gemini-{model_type}")
        self.temperature = temperature

    # ---- 与真实 SDK 的常用接口保持同名 -----------------
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """离线桩：直接返回可预测占位文本"""
        return "stub-response"

    def count_tokens(self, text: str) -> int:
        """非常粗糙的 token 估计，只用于避免 divide‑by‑zero"""
        return max(1, len(text) // 4)

    # ---- 供测试用的元信息接口 --------------------------
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "temperature": self.temperature,
        }


# ---------- 工厂函数：保持旧测试脚本接口 ----------------
def create_recommendation_agent() -> GeminiWrapper:   # noqa
    return GeminiWrapper(model_type="flash", temperature=0.7)


def create_evaluation_agent() -> GeminiWrapper:       # noqa
    return GeminiWrapper(model_type="pro", temperature=0.3)


def create_optimizer_agent() -> GeminiWrapper:        # noqa
    return GeminiWrapper(model_type="flash-thinking", temperature=0.5)
