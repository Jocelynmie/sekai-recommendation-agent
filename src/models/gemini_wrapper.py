import asyncio
import time
from typing import Dict, List, Optional, Any, Union
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

@dataclass
class GeminiResponse:
    """Response from Gemini API"""
    content: str
    success: bool
    model_used: str
    tokens_used: Optional[int] = None
    response_time: float = 0.0
    error_message: Optional[str] = None

class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a request"""
        async with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < 60]
            
            if len(self.requests) >= self.max_requests:
                # Wait until we can make another request
                sleep_time = 60 - (now - self.requests[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            self.requests.append(now)

class GeminiWrapper:
    """Wrapper for Google Gemini API with support for multiple models"""
    
    def __init__(self, api_key: str, 
                 default_model: str = "gemini-2.0-flash-exp",
                 max_requests_per_minute: int = 60):
        """
        Initialize Gemini wrapper
        
        Args:
            api_key: Google API key
            default_model: Default model to use
            max_requests_per_minute: Rate limit for API calls
        """
        self.api_key = api_key
        self.default_model = default_model
        self.rate_limiter = RateLimiter(max_requests_per_minute)
        
        # Configure Gemini
        genai.configure(api_key=api_key)  # type: ignore
        
        # Available models
        self.models = {
            "gemini-2.0-flash-exp": {
                "name": "gemini-2.0-flash-exp",
                "description": "Fast model for recommendations and simple tasks",
                "max_tokens": 8192,
                "temperature": 0.7
            },
            "gemini-1.5-pro": {
                "name": "gemini-1.5-pro", 
                "description": "Advanced model for reasoning and complex tasks",
                "max_tokens": 32768,
                "temperature": 0.3
            }
        }
        
        # Initialize model instances
        self.model_instances = {}
        for model_name in self.models:
            try:
                self.model_instances[model_name] = genai.GenerativeModel( # type: ignore
                    model_name=model_name,
                    generation_config=genai.types.GenerationConfig( # type: ignore
                        temperature=self.models[model_name]["temperature"],
                        max_output_tokens=self.models[model_name]["max_tokens"]
                    ),
                    safety_settings=[
                        {
                            "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                            "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                        },
                        {
                            "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                        },
                        {
                            "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                        },
                        {
                            "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                        }
                    ]
                ) 
            except Exception as e:
                logging.warning(f"Failed to initialize model {model_name}: {e}") # type: ignore
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Gemini wrapper initialized with models: {list(self.model_instances.keys())}")
    
    async def generate(self, 
                      prompt: str,
                      model: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      **kwargs) -> GeminiResponse:
        """
        Generate response from Gemini API
        
        Args:
            prompt: Input prompt
            model: Model to use (defaults to self.default_model)
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            GeminiResponse with results
        """
        start_time = time.time()
        model_name = model or self.default_model
        
        if model_name not in self.model_instances:
            return GeminiResponse(
                content="",
                success=False,
                model_used=model_name,
                error_message=f"Model {model_name} not available"
            )
        
        try:
            # Rate limiting
            await self.rate_limiter.acquire()
            
            # Get model instance
            model_instance = self.model_instances[model_name]
            
            # Update generation config if parameters provided
            if temperature is not None or max_tokens is not None:
                current_config = model_instance.generation_config
                new_config = genai.types.GenerationConfig( # type: ignore
                    temperature=temperature if temperature is not None else current_config.temperature,
                    max_output_tokens=max_tokens if max_tokens is not None else current_config.max_output_tokens
                )
                model_instance = genai.GenerativeModel( # type: ignore
                    model_name=model_name,
                    generation_config=new_config,
                    safety_settings=model_instance.safety_settings
                )
            
            # Generate response
            response = await asyncio.to_thread(
                model_instance.generate_content,
                prompt,
                **kwargs
            )
            
            response_time = time.time() - start_time
            
            # Extract content and token usage
            content = response.text if response.text else ""
            tokens_used = None
            if hasattr(response, 'usage_metadata'):
                tokens_used = response.usage_metadata.total_token_count
            
            return GeminiResponse(
                content=content,
                success=True,
                model_used=model_name,
                tokens_used=tokens_used,
                response_time=response_time
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error(f"Error generating response with {model_name}: {e}")
            
            return GeminiResponse(
                content="",
                success=False,
                model_used=model_name,
                response_time=response_time,
                error_message=str(e)
            )
    
    async def generate_batch(self,
                           prompts: List[str],
                           model: Optional[str] = None,
                           max_concurrent: int = 5,
                           **kwargs) -> List[GeminiResponse]:
        """
        Generate responses for multiple prompts concurrently
        
        Args:
            prompts: List of prompts to process
            model: Model to use
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional generation parameters
            
        Returns:
            List of GeminiResponse objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_single(prompt: str) -> GeminiResponse:
            async with semaphore:
                return await self.generate(prompt, model=model, **kwargs)
        
        tasks = [generate_single(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True) # type: ignore
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        return self.models.get(model_name)
    
    def list_models(self) -> List[str]:
        """List available models"""
        return list(self.model_instances.keys())
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "rate_limiter": {
                "max_requests_per_minute": self.rate_limiter.max_requests,
                "current_requests": len(self.rate_limiter.requests)
            },
            "available_models": self.list_models(),
            "default_model": self.default_model
        }

# Convenience functions for common use cases
async def generate_recommendation(prompt: str, 
                                api_key: str,
                                **kwargs) -> GeminiResponse:
    """Generate recommendation using fast model"""
    wrapper = GeminiWrapper(api_key, default_model="gemini-2.0-flash-exp")
    return await wrapper.generate(prompt, **kwargs)

async def generate_reasoning(prompt: str,
                           api_key: str,
                           **kwargs) -> GeminiResponse:
    """Generate reasoning using advanced model"""
    wrapper = GeminiWrapper(api_key, default_model="gemini-1.5-pro")
    return await wrapper.generate(prompt, **kwargs)
