import json
import os
from typing import List, Tuple

class KnowledgeGraph:
    """Wrapper around python-igraph for HippoRAG."""
    
    def __init__(self):
        try:
            import igraph as ig
        except ImportError:
            raise ImportError("Please install igraph to use KnowledgeGraph.")
            
        self.ig = ig
        self.graph = self.ig.Graph(directed=True)
        # We need a mapping from string node names to integer IDs used by igraph
        self.node_to_id = {}
        
    def add_node(self, node_name: str) -> int:
        if node_name not in self.node_to_id:
            vid = self.graph.vcount()
            self.graph.add_vertices(1)
            self.graph.vs[vid]["name"] = node_name
            self.node_to_id[node_name] = vid
        return self.node_to_id[node_name]
        
    def add_edge(self, source: str, target: str, relation: str = ""):
        u = self.add_node(source)
        v = self.add_node(target)
        self.graph.add_edges([(u, v)])
        edge_id = self.graph.ecount() - 1
        self.graph.es[edge_id]["relation"] = relation
        
    def build_from_triples(self, triples_list: List[Tuple[str, str, str]]):
        """Bulk builds graph from a list of (subject, relation, object) tuples."""
        # Using add_vertices and add_edges in bulk is faster
        new_nodes = set()
        edges_to_add = []
        relations = []
        
        for subj, rel, obj in triples_list:
            if subj not in self.node_to_id:
                new_nodes.add(subj)
            if obj not in self.node_to_id:
                new_nodes.add(obj)
                
        # Add all new nodes
        for node in new_nodes:
            self.add_node(node)
            
        for subj, rel, obj in triples_list:
            edges_to_add.append((self.node_to_id[subj], self.node_to_id[obj]))
            relations.append(rel)
            
        self.graph.add_edges(edges_to_add)
        self.graph.es[-len(edges_to_add):]["relation"] = relations

    def get_nodes(self) -> List[str]:
        return list(self.node_to_id.keys())
        
    def get_node_id(self, node_name: str) -> int:
        return self.node_to_id.get(node_name, -1)

    def save(self, file_path: str):
        self.graph.write_graphml(file_path)
        
    def load(self, file_path: str):
        self.graph = self.ig.Graph.Read_GraphML(file_path)
        self.node_to_id = {v["name"]: v.index for v in self.graph.vs}
