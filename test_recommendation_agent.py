#!/usr/bin/env python3
"""
Test script for RecommendationAgent using Gemini 2.0 Flash
"""

import asyncio
import os
import sys

# Add src to path
sys.path.append('src')

from agents.recommendation import RecommendationAgent, get_story_recommendations


async def test_recommendation_agent():
    """Test the recommendation agent with sample user tags"""
    
    # Check if API key is available
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ GOOGLE_API_KEY environment variable not set")
        print("Please set your Google API key: export GOOGLE_API_KEY='your-api-key'")
        return
    
    print("🚀 Testing RecommendationAgent with Gemini 2.0 Flash")
    print("=" * 60)
    
    # Sample user tags for testing
    test_tags = [
        "romance, anime, my hero academia, bakugo",
        "naruto, action, adventure, ninja",
        "supernatural, demon slayer, fantasy",
        "harem, reverse harem, love triangle"
    ]
    
    try:
        # Create agent
        agent = RecommendationAgent(api_key=api_key)
        print(f"✅ Agent created successfully")
        print(f"📚 Available stories: {agent.get_available_stories_count()}")
        print()
        
        # Test each set of tags
        for i, tags in enumerate(test_tags, 1):
            print(f"🧪 Test {i}: {tags}")
            print("-" * 40)
            
            # Get recommendations
            response = await agent.process_async(tags)
            
            if response.success:
                content = response.content
                print(f"✅ Success! Found {len(content['recommended_story_ids'])} recommendations")
                print(f"🤖 Model used: {content['model_used']}")
                print(f"⏱️  Response time: {content['response_time']:.2f}s")
                print(f"🔢 Tokens used: {content['tokens_used']}")
                print()
                
                # Show top 3 recommendations
                print("📖 Top 3 Recommendations:")
                for j, story in enumerate(content['recommended_stories'][:3], 1):
                    print(f"  {j}. ID: {story['content_id']}")
                    print(f"     Title: {story['title']}")
                    print(f"     Characters: {story['characters']}")
                    print(f"     Preview: {story['intro_preview'][:100]}...")
                    print()
            else:
                print(f"❌ Error: {response.error_message}")
                print()
            
            print("=" * 60)
            print()
        
        # Test the quick helper function
        print("🧪 Testing quick helper function...")
        quick_result = await get_story_recommendations("romance, anime", api_key)
        if "error" not in quick_result:
            print(f"✅ Quick function works! Found {len(quick_result['recommended_story_ids'])} stories")
        else:
            print(f"❌ Quick function error: {quick_result['error']}")
        
        # Show agent stats
        print("\n📊 Agent Statistics:")
        stats = agent.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


async def test_without_api_key():
    """Test the agent without API key (should use fallback)"""
    print("🧪 Testing RecommendationAgent without API key (fallback mode)")
    print("=" * 60)
    
    try:
        # Create agent without API key
        agent = RecommendationAgent()
        print(f"✅ Agent created successfully (fallback mode)")
        print(f"📚 Available stories: {agent.get_available_stories_count()}")
        print()
        
        # Test with tags
        tags = "romance, anime, action"
        print(f"🧪 Testing with tags: {tags}")
        
        # Use synchronous process method (fallback)
        response = agent.process(tags)
        
        if response.success:
            content = response.content
            print(f"✅ Success! Found {len(content['recommended_story_ids'])} recommendations")
            print(f"🤖 Model used: {content['model_used']}")
            print()
            
            # Show top 3 recommendations
            print("📖 Top 3 Recommendations:")
            for j, story in enumerate(content['recommended_stories'][:3], 1):
                print(f"  {j}. ID: {story['content_id']}")
                print(f"     Title: {story['title']}")
                print(f"     Characters: {story['characters']}")
                print()
        else:
            print(f"❌ Error: {response.error_message}")
            
    except Exception as e:
        print(f"❌ Error during fallback testing: {e}")


if __name__ == "__main__":
    print("🎯 RecommendationAgent Test Suite")
    print("=" * 60)
    
    # Test with API key if available
    if os.getenv('GOOGLE_API_KEY'):
        asyncio.run(test_recommendation_agent())
    else:
        print("⚠️  No API key found, testing fallback mode only")
        asyncio.run(test_without_api_key())
    
    print("\n✨ Test completed!") 