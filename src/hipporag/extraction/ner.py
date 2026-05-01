from typing import List
from ..llm.base import BaseLLM

class QueryNERExtractor:
    """Handles extracting Named Entities from user queries for online retrieval."""
    
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.prompt_template = """You’re a very effective entity extraction system. Please extract all named entities that are important for solving the questions below. Place the named entities in JSON format.

One-Shot Demonstration:
Question: Which magazine was started first Arthur’s Magazine or First for Women?
{{
    "named_entities": ["First for Women", "Arthur’s Magazine"]
}}

Input:
Question: {query}
"""

    def extract(self, query: str) -> List[str]:
        prompt = self.prompt_template.format(query=query)
        try:
            result = self.llm.extract_json(prompt)
            return result.get('named_entities', [])
        except Exception as e:
            print(f"Query NER Extraction failed for query '{query}'. Error: {e}")
            return []
