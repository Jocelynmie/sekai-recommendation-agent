#!/usr/bin/env python3
"""
Simple example of using the RecommendationAgent
"""

import asyncio
import os
import sys

# Add src to path
sys.path.append('src')

from agents.recommendation import RecommendationAgent


async def main():
    """Example usage of RecommendationAgent"""
    
    # Set your API key (or use environment variable)
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Please set GOOGLE_API_KEY environment variable")
        return
    
    # Create the agent
    agent = RecommendationAgent(api_key=api_key)
    
    # Example user tags
    user_tags = "romance, anime, my hero academia, bakugo"
    
    print(f"🎯 Getting recommendations for: {user_tags}")
    print("=" * 50)
    
    # Get recommendations
    response = await agent.process_async(user_tags)
    
    if response.success:
        recommendations = response.content
        
        print(f"✅ Found {len(recommendations['recommended_story_ids'])} recommendations")
        print(f"🤖 Using model: {recommendations['model_used']}")
        print(f"⏱️  Response time: {recommendations['response_time']:.2f}s")
        print()
        
        # Display recommendations
        for i, story in enumerate(recommendations['recommended_stories'], 1):
            print(f"{i}. {story['title']}")
            print(f"   ID: {story['content_id']}")
            print(f"   Characters: {story['characters']}")
            print(f"   Preview: {story['intro_preview'][:150]}...")
            print()
    else:
        print(f"❌ Error: {response.error_message}")


if __name__ == "__main__":
    asyncio.run(main()) 