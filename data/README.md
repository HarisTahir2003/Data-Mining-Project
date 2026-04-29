# Datasets

## Source

All files were pulled from the **HippoRAG GitHub repo, `legacy` branch (= `v1.0.0` tag)**, which is the snapshot used for the NeurIPS 2024 paper:

```
git clone --depth 1 --branch legacy --filter=blob:none --no-checkout \
    https://github.com/OSU-NLP-Group/HippoRAG.git
git -C HippoRAG sparse-checkout init --cone
git -C HippoRAG sparse-checkout set data
git -C HippoRAG checkout legacy
```

We then copied 10 files (the dev sets the paper evaluates on, plus the small training subsets the authors used to tune HippoRAG's two hyperparameters per Section 3.4) into `raw/`.

## Folder layout

```
data/
├── raw/          # original JSON files, never modified
└── processed/    # pandas-friendly Parquet/CSV builds (populated later)
```

## Files in `raw/`

| File | Records | Type | Schema (top-level keys) | Notes |
|---|---:|---|---|---|
| `musique.json`              | 1,000 dev Qs | list[dict]  | `id`, `question`, `answer`, `answer_aliases`, `answerable`, `paragraphs` | Each `paragraph` carries an `is_supporting` flag |
| `musique_corpus.json`       | 11,656 passages | list[dict] | `title`, `text` | Concatenation of every candidate passage from the 1k Qs |
| `musique_train.json`        | 100 train Qs | list[dict] | same as dev | Used by the authors only for hyperparam tuning |
| `musique_train_corpus.json` | 1,890 passages | list[dict] | same as dev | |
| `2wikimultihopqa.json`            | 1,000 dev Qs | list[dict] | `_id`, `question`, `answer`, `answer_id`, `context`, `entity_ids`, `evidences`, … | 2Wiki carries explicit per-hop entity IDs |
| `2wikimultihopqa_corpus.json`     | 6,119 passages | list[dict]   | `title`, `text` | |
| `hotpotqa.json`             | 1,000 dev Qs | list[dict]   | `_id`, `question`, `answer`, `context`, `supporting_facts`, `level`, `type` | `level` ∈ {easy, medium, hard}; `type` ∈ {bridge, comparison} |
| `hotpotqa_corpus.json`      | 9,221 passages | **dict** {title: list[sentence]} | — | **Different format** from the other two corpora: title → list of sentence strings rather than `[{title, text}, …]`. Loaders need to handle this |
| `hotpotqa_train.json`       | 100 train Qs | list[dict] | same as dev | |
| `hotpotqa_train_corpus.json`| dict format  | dict | — | |

All passage counts match **Table 1** of the paper exactly:
MuSiQue 11,656 ✓ — 2WikiMultiHopQA 6,119 ✓ — HotpotQA 9,221 ✓.

## Notes for downstream EDA

- **HotpotQA's corpus is shaped differently** (dict of title→sentences vs. list of `{title, text}`). The unified loader in `notebooks/eda_real_data.ipynb` (next milestone) will normalise this.
- The `paragraphs` field in MuSiQue and the `context` field in 2Wiki/HotpotQA both contain the candidate-passage list *as it was shown to the question's annotator*, with `is_supporting` (MuSiQue) or `supporting_facts` (HotpotQA) marking the gold passages. Distractors are everything else in that list.
- No 2WikiMultiHopQA train file exists in the source repo — only MuSiQue and HotpotQA have them.

## How to re-download

If `raw/` is ever deleted, re-running the sparse-checkout block at the top will regenerate it. Total size ≈ 44 MB.
