# HippoRAG: Implementation and Evaluation

Data mining course project reimplementing **HippoRAG**, a retrieval-augmented
generation method that builds a knowledge graph over a corpus and retrieves across
it using personalised PageRank, rather than treating each chunk independently.

## What is here

```
src/hipporag/
  data/         corpus loading and chunking
  extraction/   open information extraction into triples
  graph/        knowledge graph construction and PageRank retrieval
  embeddings/   passage and node embedding
  retrieval/    the retrieval pipeline
  qa/           question answering over retrieved context
  eval/         evaluation metrics
notebooks/
  HippoRAG_Pipeline.ipynb   end-to-end run
  eda_hippo_rag.ipynb       analysis of the graph and retrieval behaviour
  eda_real_data.ipynb       exploratory analysis of the corpus
main.tex                    final report, NeurIPS format
HippoRAG.pdf                the original paper
Data_Mining_Project_intermediate_report.pdf
```

## Getting started

```bash
pip install -r src/requirements.txt
jupyter notebook notebooks/HippoRAG_Pipeline.ipynb
```
