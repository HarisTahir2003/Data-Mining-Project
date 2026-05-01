import re
import string
from typing import Set

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
        
    return white_space_fix(remove_articles(remove_punc(lower(s))))

class Evaluator:
    @staticmethod
    def exact_match(prediction: str, ground_truth: str) -> bool:
        return normalize_answer(prediction) == normalize_answer(ground_truth)
        
    @staticmethod
    def recall_at_k(retrieved_titles: Set[str], gold_titles: Set[str]) -> bool:
        """Returns True if ALL gold titles are present in the retrieved titles."""
        return gold_titles.issubset(retrieved_titles)
