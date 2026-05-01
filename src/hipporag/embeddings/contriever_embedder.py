import torch
import numpy as np
from typing import List, Union
from .base import BaseEmbedder

class ContrieverEmbedder(BaseEmbedder):
    """Implementation for facebook/contriever using HuggingFace Transformers."""
    
    def __init__(self, model_name: str = "facebook/contriever"):
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError("Please install transformers and torch to use ContrieverEmbedder.")
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
            
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
                all_embeddings.append(embeddings.cpu().numpy())
                
        return np.vstack(all_embeddings)
