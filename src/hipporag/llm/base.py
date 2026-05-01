from abc import ABC, abstractmethod
import json

class BaseLLM(ABC):
    """Abstract Base Class for Large Language Models used in HippoRAG."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generates text from the given prompt."""
        pass
        
    def extract_json(self, prompt: str) -> dict | list:
        """
        Generates text and safely extracts JSON from the output.
        Automatically removes markdown formatting and conversational filler.
        """
        response_text = self.generate(prompt).strip()
        
        # Use regex to find the outermost JSON object or array
        import re
        match = re.search(r'(\{.*\}|\[.*\])', response_text, re.DOTALL)
        
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Failed to parse extracted JSON. Raw output: {response_text}")
                raise e
        else:
            print(f"No JSON found in response. Raw output: {response_text}")
            raise ValueError("No JSON block found in the LLM response.")
