from abc import ABC, abstractmethod
import numpy as np
from typing import List, Union

class BaseEmbedder(ABC):
    """Abstract Base Class for Dense Retrieval Embedders."""
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Encodes a list of texts into a 2D numpy array of embeddings."""
        pass
