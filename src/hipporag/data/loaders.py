import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Set

@dataclass
class Passage:
    title: str
    text: str
    id: Optional[str] = None

@dataclass
class Query:
    id: str
    question: str
    answer: str
    gold_titles: Set[str]
    retrieved_passages: List[Passage] = None

class DatasetLoader:
    """Base class for loading different datasets."""
    @staticmethod
    def load_hotpotqa(queries_path: str, corpus_path: str, limit: int = None) -> tuple[List[Query], Dict[str, Passage]]:
        """
        Loads HotpotQA style queries and corpus.
        Returns a tuple of (queries, corpus_dict)
        """
        with open(queries_path, 'r') as f:
            raw_queries = json.load(f)
            
        if limit:
            raw_queries = raw_queries[:limit]
            
        required_titles = set()
        for q in raw_queries:
            if 'context' in q:
                for ctx in q['context']:
                    required_titles.add(ctx[0])
                    
        with open(corpus_path, 'r') as f:
            raw_corpus = json.load(f)
            
        corpus = {}
        for title, sentences in raw_corpus.items():
            if title in required_titles:
                if isinstance(sentences, list):
                    text = " ".join(sentences)
                else:
                    text = str(sentences)
                corpus[title] = Passage(title=title, text=text, id=title)
            
        queries = []
        for q in raw_queries:
            gold_titles = set()
            if 'supporting_facts' in q:
                for sf in q['supporting_facts']:
                    gold_titles.add(sf[0])
                    
            queries.append(Query(
                id=q.get('_id', ''),
                question=q.get('question', ''),
                answer=q.get('answer', ''),
                gold_titles=gold_titles
            ))
            
        return queries, corpus

    @staticmethod
    def load_musique(dataset_path: str, limit: int = None) -> tuple[List[Query], Dict[str, Passage]]:
        """
        Loads MuSiQue style dataset.
        MuSiQue comes with paragraphs embedded in the query objects.
        Returns a tuple of (queries, corpus_dict)
        """
        with open(dataset_path, 'r') as f:
            raw_data = json.load(f)
            
        if limit:
            raw_data = raw_data[:limit]
            
        corpus = {}
        queries = []
        
        for q in raw_data:
            gold_titles = set()
            retrieved_passages = []
            
            for p in q.get('paragraphs', []):
                title = p['title']
                text = p['paragraph_text']
                if title not in corpus:
                    corpus[title] = Passage(title=title, text=text, id=title)
                
                if p.get('is_supporting', False):
                    gold_titles.add(title)
                    
            queries.append(Query(
                id=q.get('id', ''),
                question=q.get('question', ''),
                answer=q.get('answer', ''), # Usually string or list
                gold_titles=gold_titles
            ))
            
        return queries, corpus

    @staticmethod
    def load_2wikimultihopqa(queries_path: str, corpus_path: str, limit: int = None) -> tuple[List[Query], Dict[str, Passage]]:
        """
        Loads 2WikiMultiHopQA style queries and corpus.
        Returns a tuple of (queries, corpus_dict)
        """
        with open(queries_path, 'r') as f:
            raw_queries = json.load(f)
            
        if limit:
            raw_queries = raw_queries[:limit]
            
        required_titles = set()
        for q in raw_queries:
            if 'context' in q:
                for ctx in q['context']:
                    required_titles.add(ctx[0])

        with open(corpus_path, 'r') as f:
            raw_corpus = json.load(f)
            
        corpus = {}
        for item in raw_corpus:
            title = item['title']
            if title in required_titles:
                text = item['text']
                corpus[title] = Passage(title=title, text=text, id=title)
            
        queries = []
        for q in raw_queries:
            gold_titles = set()
            if 'supporting_facts' in q:
                for sf in q['supporting_facts']:
                    gold_titles.add(sf[0])
                    
            queries.append(Query(
                id=q.get('_id', ''),
                question=q.get('question', ''),
                answer=q.get('answer', ''),
                gold_titles=gold_titles
            ))
            
        return queries, corpus
