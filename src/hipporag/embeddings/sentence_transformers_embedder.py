import numpy as np
from typing import List, Union
from .base import BaseEmbedder

class SentenceTransformerEmbedder(BaseEmbedder):
    """Wrapper for any sentence-transformers model (e.g. all-MiniLM-L6-v2, NV-Embed-v2)."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install sentence-transformers to use SentenceTransformerEmbedder.")
            
        self.model = SentenceTransformer(model_name)
        
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
