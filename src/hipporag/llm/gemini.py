import os
import time
from .base import BaseLLM

class GeminiLLM(BaseLLM):
    """Implementation of BaseLLM using the Google Generative AI SDK."""
    
    def __init__(self, model_name: str = "gemini-flash-lite-latest", rate_limit_sleep: int = 5):
        try:
            from google import genai
        except ImportError:
            raise ImportError("Please install google-genai to use GeminiLLM.")
            
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required.")
            
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.rate_limit_sleep = rate_limit_sleep
        
    def generate(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            if self.rate_limit_sleep > 0:
                time.sleep(self.rate_limit_sleep)
            return response.text
        except Exception as e:
            print(f"Gemini API Error: {e}")
            if self.rate_limit_sleep > 0:
                time.sleep(self.rate_limit_sleep)
            return ""
