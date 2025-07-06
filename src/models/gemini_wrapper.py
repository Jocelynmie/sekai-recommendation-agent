"""
Gemini Model Wrapper
Offline stub implementation for testing
"""

import json
import time
from typing import Optional, Dict, Any
from .model_wrapper import BaseModelWrapper


class GeminiWrapper(BaseModelWrapper):
    """Minimal implementation to pass tests"""
    
    def __init__(self, model_name: str = "gemini-2.0-flash-exp", temperature: float = 0.7):
        super().__init__(model_name, temperature)
        self.model_name = model_name
        self.temperature = temperature
        self.request_count = 0
        
        # ---- Keep same interface as real SDK ----
        self.system_prompt = "You are a helpful AI assistant."
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Offline stub: directly return predictable placeholder text"""
        self.request_count += 1
        
        # Simulate processing time
        time.sleep(0.1)
        
        # Return mock response based on prompt content
        if "recommend" in prompt.lower():
            return "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
        elif "evaluate" in prompt.lower():
            return "5"
        else:
            return f"Mock Gemini response to: {prompt[:50]}..."
    
    def count_tokens(self, text: str) -> int:
        """Very rough token estimation, only to avoid divide-by-zero"""
        return len(text) // 4
    
    def clear_cache(self):
        """Clear model cache"""
        pass
    
    # ---- Meta information interface for testing ----
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "request_count": self.request_count
        }


# ---------- Factory function: maintain old test script interface ----------------
def create_gemini_agent(model_name: str = "gemini-2.0-flash-exp") -> GeminiWrapper:
    """Create Gemini agent instance"""
    return GeminiWrapper(model_name)
