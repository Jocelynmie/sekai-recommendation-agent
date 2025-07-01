import asyncio
import json
import os
import time
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime

from .base import BaseAgent, AgentResponse
from ..models.gemini_wrapper import GeminiWrapper


class RecommendationAgent(BaseAgent):
    """Agent for recommending stories using Gemini 2.0 Flash"""
    
    def __init__(self, 
                 agent_id: str = "recommendation_agent",
                 name: str = "Story Recommendation Agent",
                 description: str = "Uses Gemini 2.0 Flash to recommend stories based on user tags",
                 api_key: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize recommendation agent
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name
            description: Agent description
            api_key: Google API key for Gemini
            config: Configuration dictionary
        """
        super().__init__(agent_id, name, description, config)
        
        # Get API key from config or environment
        self.api_key = api_key or self.config.get('api_key') or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize Gemini wrapper
        self.gemini = GeminiWrapper(
            api_key=self.api_key,
            default_model="gemini-2.0-flash-exp",
            max_requests_per_minute=60
        )
        
        # Load story data
        self.stories_df = self._load_stories_data()
        
        # Recommendation prompt template
        self.prompt_template = self._get_prompt_template()
        
        self.logger.info(f"RecommendationAgent initialized with {len(self.stories_df)} stories")
    
    def _load_stories_data(self) -> pd.DataFrame:
        """Load stories data from CSV file"""
        try:
            data_path = os.path.join("data", "raw", "contents.csv")
            df = pd.read_csv(data_path)
            self.logger.info(f"Loaded {len(df)} stories from {data_path}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load stories data: {e}")
            raise
    
    def _get_prompt_template(self) -> str:
        """Get the prompt template for story recommendations"""
        return """You are a story recommendation expert. Based on the user's interest tags, recommend exactly 10 story IDs that would be most appealing to them.

User Tags: {user_tags}

Available Stories (format: ID - Title - Characters - Brief Description):
{stories_summary}

Instructions:
1. Analyze the user's tags to understand their preferences
2. Select exactly 10 story IDs that best match their interests
3. Consider character preferences, themes, and story types
4. Return ONLY a JSON array of 10 story IDs as integers, no explanations
5. Example format: [1748, 213977, 213987, ...]

Recommended Story IDs:"""
    
    def _prepare_stories_summary(self, max_stories: int = 50) -> str:
        """Prepare a summary of available stories for the prompt"""
        # Sample stories to include in prompt (limit to avoid token limits)
        sample_stories = self.stories_df.head(max_stories)
        
        stories_summary = []
        for _, story in sample_stories.iterrows():
            # Truncate intro to keep prompt manageable
            intro_preview = str(story['intro'])[:100] + "..." if len(str(story['intro'])) > 100 else str(story['intro'])
            
            summary = f"{story['content_id']} - {story['title']} - {story['character_list']} - {intro_preview}"
            stories_summary.append(summary)
        
        return "\n".join(stories_summary)
    
    async def process_async(self, input_data: Any, **kwargs) -> AgentResponse:
        """
        Async version of process method for recommendation agent
        
        Args:
            input_data: User tags as string or list
            **kwargs: Additional arguments
            
        Returns:
            AgentResponse with recommendations and metadata
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing async recommendation request with {type(input_data).__name__}")
            
            # Convert input to string if it's a list
            if isinstance(input_data, list):
                user_tags = ", ".join(input_data)
            else:
                user_tags = str(input_data)
            
            self.logger.info(f"Processing recommendation request for tags: {user_tags}")
            
            # Prepare prompt
            stories_summary = self._prepare_stories_summary()
            prompt = self.prompt_template.format(
                user_tags=user_tags,
                stories_summary=stories_summary
            )
            
            # Generate recommendation using Gemini
            response = await self.gemini.generate(
                prompt=prompt,
                model="gemini-2.0-flash-exp",
                temperature=0.3,  # Lower temperature for more consistent recommendations
                max_tokens=500
            )
            
            if not response.success:
                raise Exception(f"Gemini API error: {response.error_message}")
            
            # Parse response to extract story IDs
            recommended_ids = self._parse_recommendation_response(response.content)
            
            # Get full story details for recommended stories
            recommended_stories = self._get_story_details(recommended_ids)
            
            # Track tokens used
            if response.tokens_used:
                self._track_tokens(response.tokens_used, "story_recommendation")
            
            result = {
                "recommended_story_ids": recommended_ids,
                "recommended_stories": recommended_stories,
                "user_tags": user_tags,
                "total_stories_available": len(self.stories_df),
                "model_used": response.model_used,
                "tokens_used": response.tokens_used,
                "response_time": response.response_time
            }
            
            response_time = time.time() - start_time
            self._track_request(True, response_time)
            
            return AgentResponse(
                content=result,
                success=True,
                metadata={'agent_id': self.agent_id, 'agent_name': self.name},
                response_time=response_time
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            self._track_request(False, response_time)
            
            self.logger.error(f"Error processing async request: {str(e)}", exc_info=True)
            
            return AgentResponse(
                content=None,
                success=False,
                error_message=str(e),
                metadata={'agent_id': self.agent_id, 'agent_name': self.name},
                response_time=response_time
            )
    
    def _process_impl(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Implementation of recommendation processing (synchronous fallback)
        
        Args:
            input_data: User tags as string or list
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with recommended story IDs and metadata
        """
        # For synchronous processing, we'll use a simple rule-based approach
        # Convert input to string if it's a list
        if isinstance(input_data, list):
            user_tags = ", ".join(input_data)
        else:
            user_tags = str(input_data)
        
        self.logger.info(f"Processing sync recommendation request for tags: {user_tags}")
        
        # Simple fallback: return first 10 stories
        recommended_ids = self.stories_df['content_id'].head(10).tolist()
        recommended_stories = self._get_story_details(recommended_ids)
        
        return {
            "recommended_story_ids": recommended_ids,
            "recommended_stories": recommended_stories,
            "user_tags": user_tags,
            "total_stories_available": len(self.stories_df),
            "model_used": "fallback",
            "tokens_used": 0,
            "response_time": 0.0
        }
    
    def _parse_recommendation_response(self, response_text: str) -> List[int]:
        """Parse Gemini response to extract story IDs"""
        try:
            # Try to find JSON array in the response
            import re
            
            # Look for JSON array pattern
            json_match = re.search(r'\[[\d,\s]+\]', response_text)
            if json_match:
                json_str = json_match.group()
                story_ids = json.loads(json_str)
                return [int(id) for id in story_ids if isinstance(id, (int, str)) and str(id).isdigit()]
            
            # Fallback: extract numbers that could be story IDs
            numbers = re.findall(r'\b\d{4,6}\b', response_text)  # 4-6 digit numbers
            story_ids = [int(num) for num in numbers[:10]]  # Take first 10
            
            if not story_ids:
                raise ValueError("No valid story IDs found in response")
            
            return story_ids
            
        except Exception as e:
            self.logger.error(f"Failed to parse recommendation response: {e}")
            # Fallback: return first 10 story IDs from dataset
            return self.stories_df['content_id'].head(10).tolist()
    
    def _get_story_details(self, story_ids: List[int]) -> List[Dict[str, Any]]:
        """Get full details for recommended stories"""
        stories = []
        for story_id in story_ids:
            story_data = self.stories_df[self.stories_df['content_id'] == story_id]
            if not story_data.empty:
                story = story_data.iloc[0]
                stories.append({
                    'content_id': int(story['content_id']),
                    'title': story['title'],
                    'characters': story['character_list'],
                    'intro_preview': str(story['intro'])[:200] + "..." if len(str(story['intro'])) > 200 else str(story['intro'])
                })
        
        return stories
    
    def get_available_stories_count(self) -> int:
        """Get total number of available stories"""
        return len(self.stories_df)
    
    def get_story_by_id(self, story_id: int) -> Optional[Dict[str, Any]]:
        """Get story details by ID"""
        story_data = self.stories_df[self.stories_df['content_id'] == story_id]
        if not story_data.empty:
            story = story_data.iloc[0]
            return {
                'content_id': int(story['content_id']),
                'title': story['title'],
                'characters': story['character_list'],
                'intro': story['intro']
            }
        return None


async def create_recommendation_agent(api_key: Optional[str] = None) -> RecommendationAgent:
    """Factory function to create a recommendation agent"""
    return RecommendationAgent(api_key=api_key)


async def get_story_recommendations(user_tags: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick function to get story recommendations
    
    Args:
        user_tags: User interest tags as string
        api_key: Google API key (optional, will use environment variable if not provided)
        
    Returns:
        Dictionary with recommendations and metadata
    """
    agent = await create_recommendation_agent(api_key)
    response = await agent.process_async(user_tags)
    return response.content if response.success else {"error": response.error_message}
