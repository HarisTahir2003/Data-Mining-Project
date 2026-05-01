import numpy as np
from typing import List, Dict, Set
from ..embeddings.base import BaseEmbedder

class EntityLinker:
    """Links extracted query entities to Knowledge Graph nodes using dense embeddings."""
    
    def __init__(self, embedder: BaseEmbedder, node_names: List[str], node_embeddings: np.ndarray, threshold: float = 0.7):
        self.embedder = embedder
        self.node_names = node_names
        self.node_embeddings = node_embeddings
        self.threshold = threshold
        
        # Normalize node embeddings for fast cosine similarity
        self.node_norms = np.linalg.norm(self.node_embeddings, axis=1, keepdims=True)
        self.node_embeddings_normalized = self.node_embeddings / (self.node_norms + 1e-9)
        
    def link(self, query_entities: List[str]) -> List[str]:
        """Finds KG nodes that match the query entities above the threshold."""
        if not query_entities:
            return []
            
        q_embs = self.embedder.encode(query_entities)
        q_norms = np.linalg.norm(q_embs, axis=1, keepdims=True)
        q_embs_normalized = q_embs / (q_norms + 1e-9)
        
        # Calculate cosine similarity matrix (num_query_entities x num_nodes)
        sim_matrix = np.dot(q_embs_normalized, self.node_embeddings_normalized.T)
        
        linked_nodes = set()
        for i in range(len(query_entities)):
            best_idx = np.argmax(sim_matrix[i])
            best_sim = sim_matrix[i, best_idx]
            
            if best_sim >= self.threshold:
                linked_nodes.add(self.node_names[best_idx])
                
        return list(linked_nodes)
