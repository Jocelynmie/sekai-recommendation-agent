#!/usr/bin/env python3
"""
Test script to verify all key dependencies are properly installed
"""

def test_dependencies():
    """Test all key dependencies"""
    print("🔍 Testing key dependencies...")
    
    # Test core data processing
    try:
        import pandas as pd
        import numpy as np
        print("✅ pandas & numpy - OK")
    except ImportError as e:
        print(f"❌ pandas & numpy - FAILED: {e}")
    
    # Test LLM APIs
    try:
        import google.generativeai as genai
        print("✅ google-generativeai - OK")
    except ImportError as e:
        print(f"❌ google-generativeai - FAILED: {e}")
    
    try:
        import openai
        print("✅ openai - OK")
    except ImportError as e:
        print(f"❌ openai - FAILED: {e}")
    
    try:
        import anthropic
        print("✅ anthropic - OK")
    except ImportError as e:
        print(f"❌ anthropic - FAILED: {e}")
    
    # Test vector and search
    try:
        import sentence_transformers
        print("✅ sentence-transformers - OK")
    except ImportError as e:
        print(f"❌ sentence-transformers - FAILED: {e}")
    
    try:
        import faiss
        print("✅ faiss-cpu - OK")
    except ImportError as e:
        print(f"❌ faiss-cpu - FAILED: {e}")
    
    try:
        import whoosh
        print("✅ whoosh - OK")
    except ImportError as e:
        print(f"❌ whoosh - FAILED: {e}")
    
    # Test logging and config
    try:
        import loguru
        print("✅ loguru - OK")
    except ImportError as e:
        print(f"❌ loguru - FAILED: {e}")
    
    try:
        import dotenv
        print("✅ python-dotenv - OK")
    except ImportError as e:
        print(f"❌ python-dotenv - FAILED: {e}")
    
    # Test utilities
    try:
        import tqdm
        print("✅ tqdm - OK")
    except ImportError as e:
        print(f"❌ tqdm - FAILED: {e}")
    
    print("\n🎉 Dependency test completed!")

if __name__ == "__main__":
    test_dependencies()
 