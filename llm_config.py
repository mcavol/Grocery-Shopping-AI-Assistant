"""
LLM configuration for Mistral API integration - STRICT API CONNECTION ONLY
NO DEMO MODE - NO HARDCODED RESPONSES
"""

import os
import time
from typing import Optional, List
from langchain.llms.base import BaseLLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import Generation, LLMResult
import logging
import requests
import json

logger = logging.getLogger(__name__)


class MistralLLM(BaseLLM):
    """Custom Mistral LLM implementation - REQUIRES VALID API KEY."""
    
    model: str = "mistral-small-latest"
    temperature: float = 0.3
    max_tokens: Optional[int] = 800
    api_key: Optional[str] = None
    base_url: str = "https://api.mistral.ai/v1/chat/completions"
    last_request_time: float = 0
    min_request_interval: float = 1.2  # 1.2 seconds between requests for rate limiting
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "❌ MISTRAL_API_KEY is required! No demo mode available.\n"
                "Please set your API key: export MISTRAL_API_KEY='your_key_here'\n"
                "Get your key from: https://console.mistral.ai/"
            )
        
        logger.info("✅ Mistral API key configured - ready for API calls")
    
    @property
    def _llm_type(self) -> str:
        return "mistral"
    
    def _rate_limit_wait(self):
        """Rate limiting to prevent API errors."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            logger.info(f"⏱️ Rate limiting: waiting {sleep_time:.1f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _call(
        self,
        prompt: str,
        stop: Optional[list] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Make HTTP request to Mistral API - NO FALLBACKS."""
        
        if not self.api_key:
            raise ValueError("❌ MISTRAL_API_KEY is required for all operations")
        
        try:
            # Rate limiting
            self._rate_limit_wait()
            
            # Prepare request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            logger.info(f"🚀 Making Mistral API call - Model: {self.model}")
            logger.debug(f"Prompt preview: {prompt[:100]}...")
            
            # Make the request
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            # Handle response
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                logger.info("✅ Mistral API call successful")
                logger.debug(f"Response preview: {content[:100]}...")
                return content
            
            elif response.status_code == 401:
                error_msg = (
                    "❌ Mistral API: Invalid API key\n"
                    "💡 Check your MISTRAL_API_KEY environment variable\n"
                    "💡 Get a valid key from: https://console.mistral.ai/"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            elif response.status_code == 429:
                error_msg = (
                    "❌ Mistral API: Rate limit exceeded\n"
                    "💡 Please wait a few minutes and try again\n"
                    "💡 Consider upgrading your API plan for higher limits"
                )
                logger.error(error_msg)
                raise Exception(error_msg)
            
            elif response.status_code == 402:
                error_msg = (
                    "❌ Mistral API: Insufficient credits\n"
                    "💡 Please add credits to your Mistral account\n"
                    "💡 Check your balance at: https://console.mistral.ai/"
                )
                logger.error(error_msg)
                raise Exception(error_msg)
            
            elif response.status_code == 400:
                error_msg = f"❌ Mistral API: Bad request - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            else:
                error_msg = f"❌ Mistral API error {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
        except requests.exceptions.Timeout:
            error_msg = "❌ Mistral API: Request timeout (30s). Please try again."
            logger.error(error_msg)
            raise Exception(error_msg)
            
        except requests.exceptions.ConnectionError:
            error_msg = "❌ Mistral API: Connection error. Check your internet connection."
            logger.error(error_msg)
            raise Exception(error_msg)
        
        except requests.exceptions.RequestException as e:
            error_msg = f"❌ Network error calling Mistral API: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        except Exception as e:
            if "Mistral API" in str(e):
                raise  # Re-raise Mistral-specific errors
            error_msg = f"❌ Unexpected error with Mistral API: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> LLMResult:
        """Generate responses for multiple prompts."""
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([Generation(text=text)])
        
        return LLMResult(generations=generations)


def create_llm(api_key: Optional[str] = None) -> MistralLLM:
    """Create and configure Mistral LLM instance - REQUIRES API KEY."""
    return MistralLLM(api_key=api_key)


def test_mistral_connection(api_key: Optional[str] = None) -> bool:
    """Test if Mistral API connection works - REQUIRES API KEY."""
    try:
        if not api_key and not os.getenv("MISTRAL_API_KEY"):
            print("❌ No API key provided for testing")
            return False
            
        llm = create_llm(api_key)
        test_response = llm._call("Respond with only 'Connection successful' if you can read this.")
        
        if "successful" in test_response.lower():
            print(f"✅ Mistral API Connection Test: {test_response}")
            return True
        else:
            print(f"⚠️ Unexpected response: {test_response}")
            return False
            
    except Exception as e:
        print(f"❌ Mistral API Test Failed: {str(e)}")
        return False


def validate_api_key(api_key: str) -> bool:
    """Validate API key format and basic connectivity."""
    if not api_key:
        return False
    
    # Basic format check
    if len(api_key) < 20:
        return False
    
    # Try a simple API call
    try:
        test_llm = MistralLLM(api_key=api_key)
        response = test_llm._call("Say 'OK'")
        return "ok" in response.lower()
    except:
        return False


if __name__ == "__main__":
    print("🔍 Testing Mistral API connection...")
    print("="*50)
    
    # Check for API key
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("❌ MISTRAL_API_KEY environment variable not found!")
        print("💡 Please set your API key:")
        print("   export MISTRAL_API_KEY='your_key_here'")
        print("   Or add it to your .env file")
        exit(1)
    
    # Test connection
    success = test_mistral_connection(api_key)
    if success:
        print("\n🎉 Mistral API is working correctly!")
        print("🚀 Ready to run the grocery shopping assistant")
    else:
        print("\n❌ Mistral API test failed")
        print("💡 Check your API key and account credits")
        exit(1)