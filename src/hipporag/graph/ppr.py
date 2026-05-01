import numpy as np
from typing import List, Dict
from .knowledge_graph import KnowledgeGraph

class PPRRunner:
    """Runs Personalized PageRank on the Knowledge Graph."""
    
    def __init__(self, kg: KnowledgeGraph, damping: float = 0.85):
        self.kg = kg
        self.damping = damping
        
    def run(self, seed_nodes: List[str]) -> Dict[str, float]:
        """
        Runs PPR given a list of seed nodes.
        Returns a dictionary mapping node names to their PPR scores.
        """
        if not seed_nodes:
            return {}
            
        # Get igraph internal IDs
        seed_ids = [self.kg.get_node_id(n) for n in seed_nodes if self.kg.get_node_id(n) != -1]
        
        if not seed_ids:
            return {}
            
        # Create restart distribution (uniform over seed nodes)
        reset_dist = [0.0] * self.kg.graph.vcount()
        val = 1.0 / len(seed_ids)
        for vid in seed_ids:
            reset_dist[vid] = val
            
        # Run PPR using igraph
        scores = self.kg.graph.personalized_pagerank(
            vertices=None,
            directed=True,
            damping=self.damping,
            reset=reset_dist,
            weights=None
        )
        
        # Map scores back to node names
        result = {}
        for node_name, vid in self.kg.node_to_id.items():
            result[node_name] = scores[vid]
            
        return result
