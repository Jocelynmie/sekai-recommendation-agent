#!/usr/bin/env python3
"""
API密钥测试脚本
验证所有三大AI厂商的API密钥是否正确配置
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def test_api_keys():
    """Test configuration status of all API keys"""
    
    print("🔍 Checking API key configuration...")
    print("=" * 50)
    
    # Check each API key
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Google Gemini": os.getenv("GOOGLE_API_KEY"), 
        "Anthropic Claude": os.getenv("ANTHROPIC_API_KEY")
    }
    
    all_configured = True
    
    for provider, key in api_keys.items():
        if key and key != "your_openai_api_key_here" and key != "your_google_api_key_here" and key != "your_anthropic_api_key_here":
            print(f"✅ {provider}: Configured")
            print(f"   Key prefix: {key[:10]}...")
        else:
            print(f"❌ {provider}: Not configured or using default value")
            all_configured = False
    
    print("=" * 50)
    
    if all_configured:
        print("🎉 All API keys are correctly configured!")
        print("\n📋 Model Selection Priority:")
        print("1. RecommendationAgent: Gemini 2.0 Flash (Primary)")
        print("2. EvaluationAgent: Gemini 2.5 Pro (Primary)")
        print("3. PromptOptimizerAgent: Claude 3.5 Sonnet (Primary)")
        
        print("\n🚀 Now you can run the complete experiment:")
        print("python run_experiment.py --cycles 3 --users 15 --mode llm")
        
    else:
        print("⚠️  Please configure missing API keys:")
        print("1. Copy env.template to .env")
        print("2. Fill in your API keys")
        print("3. Run this script again to verify")
    
    return all_configured

def test_model_creation():
    """测试模型创建是否正常"""
    
    print("\n🧪 测试模型创建...")
    print("=" * 50)
    
    try:
        from src.models.model_wrapper import (
            create_recommendation_agent,
            create_evaluation_agent, 
            create_optimizer_agent
        )
        
        # 测试创建各个Agent
        agents = {}
        
        try:
            agents["推荐Agent"] = create_recommendation_agent()
            print(f"✅ 推荐Agent创建成功: {agents['推荐Agent'].model_name}")
        except Exception as e:
            print(f"❌ 推荐Agent创建失败: {e}")
        
        try:
            agents["评估Agent"] = create_evaluation_agent()
            print(f"✅ 评估Agent创建成功: {agents['评估Agent'].model_name}")
        except Exception as e:
            print(f"❌ 评估Agent创建失败: {e}")
        
        try:
            agents["优化Agent"] = create_optimizer_agent()
            print(f"✅ 优化Agent创建成功: {agents['优化Agent'].model_name}")
        except Exception as e:
            print(f"❌ 优化Agent创建失败: {e}")
        
        # 测试简单生成
        if agents:
            print("\n🧪 测试模型生成...")
            for name, agent in agents.items():
                try:
                    response = agent.generate("Hello, test message.")
                    print(f"✅ {name} 生成测试成功")
                except Exception as e:
                    print(f"❌ {name} 生成测试失败: {e}")
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请确保已安装所有依赖: pip install -r requirements.txt")

if __name__ == "__main__":
    print("🚀 Sekai Recommendation Agent - API密钥测试")
    print("=" * 60)
    
    # 测试API密钥配置
    keys_ok = test_api_keys()
    
    # 如果密钥配置正确，测试模型创建
    if keys_ok:
        test_model_creation()
    
    print("\n" + "=" * 60)
    print("测试完成！") 