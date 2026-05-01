from typing import List, Tuple
from ..llm.base import BaseLLM
from ..data.loaders import Passage

class QAReader:
    """Uses an LLM to generate an answer based on retrieved passages."""
    
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.prompt_template = """You are an expert Question Answering system. 
Please answer the user's question based on the provided passages.
If the exact answer is not explicitly in the passages, provide your best reasoned guess based on the context.
Keep your final answer as concise as possible (e.g., just the entity name or phrase).

Passages:
{contexts}

Question: {query}
Answer:
"""

    def generate_answer(self, query: str, retrieved_passages: List[Passage]) -> str:
        if not retrieved_passages:
            return "Information not found"
            
        context_str = "\n".join([f"Passage {i+1}: {p.text}" for i, p in enumerate(retrieved_passages)])
        prompt = self.prompt_template.format(contexts=context_str, query=query)
        
        try:
            answer = self.llm.generate(prompt).strip()
            return answer
        except Exception as e:
            print(f"QA Reader failed: {e}")
            return ""
