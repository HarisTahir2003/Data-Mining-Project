from typing import Tuple, List
from ..llm.base import BaseLLM

class OpenIEExtractor:
    """Handles extracting Triples (Subject, Relation, Object) and Entities from passages."""
    
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.prompt_template = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists.
Respond with a JSON dict containing 'named_entities' and 'triples', with each triple representing a relationship in the RDF graph.

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.

Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
One-Shot Demonstration:
Paragraph:
```
Radio City
Radio City is India’s first private FM radio station and was started on 3 July 2001. It plays Hindi, English
and regional songs. Radio City recently forayed into New Media in May 2008 with the launch of a music
portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related
features.
```
{{
    "named_entities": ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"],
    "triples": [
        ["Radio City", "located in", "India"],
        ["Radio City", "is", "private FM radio station"],
        ["Radio City", "started on", "3 July 2001"],
        ["Radio City", "plays songs in", "Hindi"],
        ["Radio City", "plays songs in", "English"],
        ["Radio City", "forayed into", "New Media"],
        ["Radio City", "launched", "PlanetRadiocity.com"],
        ["PlanetRadiocity.com", "launched in", "May 2008"],
        ["PlanetRadiocity.com", "is", "music portal"],
        ["PlanetRadiocity.com", "offers", "news"],
        ["PlanetRadiocity.com", "offers", "videos"],
        ["PlanetRadiocity.com", "offers", "songs"]
    ]
}}

Input:
Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{text}
```
"""

    def extract(self, text: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """Returns a tuple of (entities_list, triples_list)."""
        prompt = self.prompt_template.format(text=text)
        try:
            result = self.llm.extract_json(prompt)
            entities = result.get('named_entities', [])
            triples_raw = result.get('triples', [])
            
            # Ensure valid triples
            triples = [tuple(t) for t in triples_raw if isinstance(t, list) and len(t) == 3]
            return entities, triples
        except Exception as e:
            print(f"OpenIE Extraction failed for text starting with {text[:30]}... Error: {e}")
            return [], []
