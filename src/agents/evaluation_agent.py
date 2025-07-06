"""
Evaluation Agent for Recommendation System
Evaluates recommendation quality using multiple methods
"""

from __future__ import annotations

import json
import logging
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Import sentence-transformers if available, otherwise prompt for installation
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")

# Add project root to sys.path for direct test execution
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .base import BaseAgent
from .recommendation_agent import RecommendationAgent
from src.models.model_wrapper import BaseModelWrapper

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResponse:
    """Evaluation output structure for type hints and serialization"""
    user_id: int
    precision: float
    recall: float
    method_used: str
    ground_truth: List[int]
    recommended: List[int]
    user_tags: List[str]
    model_used: str = ""
    reasoning: str = ""


# Evaluation metrics registry and registration decorator
EVALUATION_METRICS_REGISTRY = {}

def register_evaluation_metric(name):
    def decorator(fn):
        EVALUATION_METRICS_REGISTRY[name] = fn
        return fn
    return decorator


class EvaluationAgent(BaseAgent):
    """Responsible for evaluating RecommendationAgent output quality"""

    def __init__(
        self,
        users_df: pd.DataFrame,
        contents_df: pd.DataFrame,
        recommendation_agent: RecommendationAgent,
        use_llm_for_ground_truth: bool = True,
        use_llm_for_tag_simulation: bool = True,
        eval_mode: str = "llm",
        k: int = 10,
        model_wrapper=None,
        name="EvaluationAgent",
        config=None,
    ):
        super().__init__(name, model_wrapper, config)
        self.users_df = users_df
        self.contents_df = contents_df
        self.recommendation_agent = recommendation_agent
        self.use_llm_for_ground_truth = use_llm_for_ground_truth
        self.use_llm_for_tag_simulation = use_llm_for_tag_simulation
        self.eval_mode = eval_mode
        self.k = k
        
        # Initialize sentence transformer if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.sentence_transformer = None
            
        # Preprocess: extract all possible tags
        self.all_tags = self._extract_all_tags()
        
        # Initialize model for evaluation
        if model_wrapper:
            self.model = model_wrapper
        else:
            from src.models.model_wrapper import create_evaluation_agent
            self.model = create_evaluation_agent()

    # --------------------------------------------------------------------- #
    # -------------------------- Core Process Methods ----------------------------- #
    # --------------------------------------------------------------------- #
    
    def evaluate_user(self, user_id: int) -> EvaluationResponse:
        """Evaluate recommendations for a single user"""
        # Write interaction log
        self.log_interaction({"user_id": user_id}, {})
        
        # Get user tags
        user_tags = self._get_user_tags(user_id)
        
        # Generate ground truth
        ground_truth = self._generate_ground_truth(user_id, user_tags)
        
        # Get recommendation results
        recommendation_result = self.recommendation_agent.recommend(user_tags, self.k)
        recommended = recommendation_result.get("content_ids", [])
        
        # Calculate metrics
        precision, recall = self._calculate_metrics(ground_truth, recommended)
        
        # Determine method used
        method_used = self._determine_method()
        
        return EvaluationResponse(
            user_id=user_id,
            precision=precision,
            recall=recall,
            method_used=method_used,
            ground_truth=ground_truth,
            recommended=recommended,
            user_tags=user_tags,
            model_used=self.model.model_name if hasattr(self.model, 'model_name') else "unknown",
            reasoning=""
        )

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """BaseAgent standard interface implementation"""
        user_id = input_data.get("user_id")
        if user_id is None:
            raise ValueError("user_id is required in input_data")
        
        result = self.evaluate_user(user_id)
        return {
            "user_id": result.user_id,
            "precision": result.precision,
            "recall": result.recall,
            "method_used": result.method_used,
            "ground_truth": result.ground_truth,
            "recommended": result.recommended,
            "user_tags": result.user_tags,
            "model_used": result.model_used,
            "reasoning": result.reasoning
        }

    # --------------------------------------------------------------------- #
    # ---------------------------- Tag Processing ---------------------------------- #
    # --------------------------------------------------------------------- #
    
    def _extract_all_tags(self) -> List[str]:
        """Extract all possible tags from all user data"""
        all_tags = set()
        
        # Extract from user tags
        if 'tags' in self.users_df.columns:
            for tags_str in self.users_df['tags'].dropna():
                if isinstance(tags_str, str):
                    tags = [tag.strip() for tag in tags_str.split(',')]
                    all_tags.update(tags)
        
        # Extract from content tags
        if 'tags' in self.contents_df.columns:
            for tags_str in self.contents_df['tags'].dropna():
                if isinstance(tags_str, str):
                    tags = [tag.strip() for tag in tags_str.split(',')]
                    all_tags.update(tags)
        
        return sorted(list(all_tags))

    def _get_user_tags(self, user_id: int) -> List[str]:
        """Get user interest tags"""
        user_row = self.users_df[self.users_df['user_id'] == user_id]
        if user_row.empty:
            return []
        
        # Cost-saving mode: directly read sim_tags
        if 'sim_tags' in user_row.columns and not user_row['sim_tags'].iloc[0] is None:
            sim_tags_str = user_row['sim_tags'].iloc[0]
            if isinstance(sim_tags_str, str) and sim_tags_str.strip():
                return [tag.strip() for tag in sim_tags_str.split(',')]
        
        # Read original user interest tags
        tags_str = user_row['tags'].iloc[0] if 'tags' in user_row.columns else ""
        if isinstance(tags_str, str) and tags_str.strip():
            return [tag.strip() for tag in tags_str.split(',')]
        
        # Fallback: compatible with old logic
        return []

    # --------------------------------------------------------------------- #
    # ---------------------------- Ground Truth Generation ---------------------------------- #
    # --------------------------------------------------------------------- #
    
    def _generate_ground_truth(self, user_id: int, user_tags: List[str]) -> List[int]:
        """Generate ground truth recommendations for evaluation"""
        if not user_tags:
            return []
        
        if self.eval_mode == "llm" and self.use_llm_for_ground_truth:
            return self._generate_llm_ground_truth(user_tags)
        elif self.eval_mode == "vector":
            return self._generate_vector_ground_truth(user_tags)
        else:
            return self._generate_keyword_ground_truth(user_tags)
    
    def _generate_llm_ground_truth(self, user_tags: List[str]) -> List[int]:
        """Generate ground truth using LLM"""
        if not self.model:
            return self._generate_keyword_ground_truth(user_tags)
        
        # Create prompt for LLM
        prompt = f"""
        Given user interest tags: {', '.join(user_tags)}
        
        Please select the top {self.k} most relevant story IDs from the following candidates.
        Consider the user's interests and preferences.
        
        Available stories:
        {self._get_story_summaries()}
        
        Return only a JSON list of integers representing the story IDs, ordered by relevance.
        """
        
        try:
            response = self.model.generate(prompt)
            # Parse JSON response
            import json
            story_ids = json.loads(response)
            return story_ids[:self.k]
        except Exception as e:
            logger.warning(f"LLM ground truth generation failed: {e}")
            return self._generate_keyword_ground_truth(user_tags)
    
    def _generate_vector_ground_truth(self, user_tags: List[str]) -> List[int]:
        """Generate ground truth using vector similarity"""
        if not self.sentence_transformer:
            return self._generate_keyword_ground_truth(user_tags)
        
        # Calculate cosine similarity
        user_embedding = self.sentence_transformer.encode(' '.join(user_tags))
        
        # Cost-saving mode: directly use embedding cosine ranking
        similarities = []
        for _, content in self.contents_df.iterrows():
            content_text = f"{content.get('title', '')} {content.get('introduction', '')}"
            content_embedding = self.sentence_transformer.encode(content_text)
            similarity = np.dot(user_embedding, content_embedding) / (np.linalg.norm(user_embedding) * np.linalg.norm(content_embedding))
            similarities.append((content['content_id'], similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [content_id for content_id, _ in similarities[:self.k]]
    
    def _generate_keyword_ground_truth(self, user_tags: List[str]) -> List[int]:
        """Generate ground truth using keyword matching"""
        scores = []
        for _, content in self.contents_df.iterrows():
            content_tags = []
            if 'tags' in content and isinstance(content['tags'], str):
                content_tags = [tag.strip() for tag in content['tags'].split(',')]
            
            # Calculate overlap score
            overlap = len(set(user_tags) & set(content_tags))
            scores.append((content['content_id'], overlap))
        
        # Sort by overlap score and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [content_id for content_id, _ in scores[:self.k]]
    
    def _get_story_summaries(self) -> str:
        """Get story summaries for LLM prompt"""
        summaries = []
        for _, content in self.contents_df.head(50).iterrows():  # Limit to first 50 for efficiency
            title = content.get('title', '')
            intro = content.get('introduction', '')
            content_id = content['content_id']
            summaries.append(f"ID {content_id}: {title} - {intro[:100]}...")
        return '\n'.join(summaries)

    # --------------------------------------------------------------------- #
    # ---------------------------- Main Evaluation Process ---------------------------------- #
    # --------------------------------------------------------------------- #
    
    def _simulate_user_tags(self, user_id: int) -> List[str]:
        """1. Simulate user tags"""
        return self._get_user_tags(user_id)
    
    def _get_recommendation_results(self, user_tags: List[str]) -> Dict[str, Any]:
        """2. Get recommendation results"""
        return self.recommendation_agent.recommend(user_tags, self.k)
    
    def _generate_ground_truth_for_eval(self, user_id: int, user_tags: List[str]) -> List[int]:
        """3. Generate ground truth"""
        return self._generate_ground_truth(user_id, user_tags)
    
    def _calculate_evaluation_metrics(self, ground_truth: List[int], recommended: List[int]) -> Tuple[float, float]:
        """4. Calculate metrics"""
        return self._calculate_metrics(ground_truth, recommended)
    
    def _determine_evaluation_method(self) -> str:
        """5. Determine method used"""
        return self._determine_method()
    
    def _calculate_metrics(self, ground_truth: List[int], recommended: List[int]) -> Tuple[float, float]:
        """Calculate precision and recall"""
        if not ground_truth or not recommended:
            return 0.0, 0.0
        
        # Calculate intersection
        intersection = set(ground_truth) & set(recommended)
        
        # Calculate precision and recall
        precision = len(intersection) / len(recommended) if recommended else 0.0
        recall = len(intersection) / len(ground_truth) if ground_truth else 0.0
        
        return precision, recall
    
    def _determine_method(self) -> str:
        """Determine which evaluation method was used"""
        if self.eval_mode == "llm" and self.use_llm_for_ground_truth:
            return "llm_evaluation"
        elif self.eval_mode == "vector":
            return "vector_evaluation"
        else:
            return "keyword_evaluation"

    # --------------------------------------------------------------------- #
    # -------------------------- Prompt Section --------------------------- #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _system_prompt() -> str:
        """For advanced LLM to judge ground-truth"""
        return (
            "You are a senior evaluation agent for a role‑play story platform. "
            "You will analyze user preferences and story content to make precise matches. "
            "Always focus on the user's explicit interests and implied preferences. "
            "Be objective and consistent in your evaluations."
        )