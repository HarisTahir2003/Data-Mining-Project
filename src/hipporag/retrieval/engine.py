import numpy as np
from typing import List, Dict, Tuple
from ..data.loaders import Passage, Query
from ..extraction.ner import QueryNERExtractor
from ..graph.ppr import PPRRunner
from .linker import EntityLinker
from ..embeddings.base import BaseEmbedder

class RetrievalEngine:
    """Coordinates the full HippoRAG online retrieval process."""
    
    def __init__(
        self,
        ner_extractor: QueryNERExtractor,
        linker: EntityLinker,
        ppr_runner: PPRRunner,
        embedder: BaseEmbedder,
        passage_embeddings: Dict[str, np.ndarray],
        passage_triples: Dict[str, List[Tuple[str, str, str]]],
        alpha: float = 0.5
    ):
        self.ner_extractor = ner_extractor
        self.linker = linker
        self.ppr_runner = ppr_runner
        self.embedder = embedder
        self.passage_embeddings = passage_embeddings
        self.passage_triples = passage_triples
        self.alpha = alpha  # Weight for graph score vs dense score
        
    def retrieve(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        # 1. Extract Query Entities
        query_entities = self.ner_extractor.extract(query_text)
        
        # 2. Link Entities to KG Nodes
        linked_nodes = self.linker.link(query_entities)
        
        # 3. Run Personalized PageRank
        ppr_scores = self.ppr_runner.run(linked_nodes)
        
        # 4. Dense query embedding for passage similarity
        query_emb = self.embedder.encode([query_text])[0]
        query_norm = np.linalg.norm(query_emb)
        
        # 5. Score Passages
        passage_scores = {}
        for p_id, p_emb in self.passage_embeddings.items():
            # Dense Similarity Score
            p_norm = np.linalg.norm(p_emb)
            dense_score = np.dot(query_emb, p_emb) / (query_norm * p_norm + 1e-9)
            
            # Graph Score (sum of PPR scores of entities in passage)
            graph_score = 0.0
            if p_id in self.passage_triples:
                entities_in_passage = set()
                for subj, rel, obj in self.passage_triples[p_id]:
                    entities_in_passage.add(subj)
                    entities_in_passage.add(obj)
                
                for ent in entities_in_passage:
                    graph_score += ppr_scores.get(ent, 0.0)
                    
            # Combine
            final_score = self.alpha * graph_score + (1 - self.alpha) * dense_score
            passage_scores[p_id] = final_score
            
        # 6. Sort and return Top-K
        sorted_passages = sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_passages[:top_k]
