# DocuMind — a document question-answering RAG service

[![CI](https://github.com/ShashwatK27/DocuMind/actions/workflows/ci.yml/badge.svg)](https://github.com/ShashwatK27/DocuMind/actions/workflows/ci.yml)

Ask questions about a PDF and get a grounded, cited answer — or an honest "I
don't know" when the document doesn't support one. DocuMind is a retrieval-
augmented generation (RAG) service whose every hyperparameter was **chosen from
measured experiments**, not guessed. The experiment log (`experiments.md`) and
the decision log (`decision.md`) are first-class parts of this repo.

## Results

The pipeline was tuned on a single document (where page-recall saturated at 1.000)
and is now evaluated on a **four-document corpus** (the roadmap plus the
Transformer, BERT and RAG papers) with a 63-question hand-grounded golden set.

| Metric | single-doc (39 Q) | 4-doc corpus (59 Q) | What it means |
|---|---:|---:|---|
| Page Recall@5 | 1.000 | **0.983** | the right page is in the top-5 retrieved chunks |
| MRR | 0.944 | **0.897** | how highly the right page is ranked |
| Content coverage | 0.759 | **0.754** | fraction of ground-truth answer terms delivered |
| Citation leaks | 0 / 43 | — | the LLM never leaks source labels into prose |
| Unanswerable refused | 4 / 4 | — | it refuses everything it can't ground |

The multi-doc numbers come from two measured retrieval upgrades. Moving to a
4-document corpus surfaced exact-match misses (rare tokens like *Ragas*, *FAISS*,
*HNSW*) that dense retrieval dilutes:

1. **Hybrid retrieval (D-16)** — augmenting the dense pool with BM25's top lexical
   matches put those chunks back into contention (Recall@5 0.932 → 0.949). A naive
   Reciprocal-Rank Fusion variant was tried first and was a wash (it evicted
   correct dense chunks before reranking); the comparison is in
   `experiment/hybrid_vs_dense_results.csv`.
2. **Query-aware fusion (D-17)** — a diagnostic showed the reranker still buried
   two recovered chunks at ~rank 7, so the final ranking blends the reranker and
   BM25 scores (`reranker_z + 0.15·bm25_z`). Swept to the knee: Recall@5 0.949 →
   **0.983**, MRR +0.007, no regressions. Of the 3 residual misses, 2 are closed;
   the 1 remaining is a genuinely ambiguous grounding, left standing rather than
   relabelled.

Full ablations (chunk size, top-k, reranker on/off, metadata enrichment, hybrid
search, prompt A/B) are in [`experiments.md`](experiments.md); the rationale and
rejected alternatives for each choice are in [`decision.md`](decision.md).

## Pipeline

```
PDF ─► load (PyPDF) ─► chunk 500/150 + metadata enrichment ─► embed (MiniLM-L6-v2, cosine)
                                                                      │
query ─► embed ─► dense top-20 ∪ BM25 top-20 ─► cross-encoder rerank ─► fuse (rerank+BM25) ─► 1 chunk/page, top-5 ─► LLM (Prompt B)
                                                                      │
                                                        grounded answer + Python-owned citations
```

Key design decisions:

- **Metadata enrichment** — each chunk is prefixed with `Source | Page | Section`;
  measured to raise content coverage at overlap 150 (D-11).
- **Hybrid retrieval** — the dense pool is augmented with BM25's top lexical
  matches and the reranker decides, recovering exact-match hits on a
  multi-document corpus (D-16).
- **Cross-encoder reranking** over a wide dense pool — lifts MRR at every top-k (D-4).
- **Query-aware fusion** — the final ranking blends the reranker and BM25 scores
  (`reranker_z + 0.15·bm25_z`) so a chunk both signals agree on reaches the
  top-5; swept to the knee, closing exact-match misses the reranker alone buried (D-17).
- **Python owns citations** — the model is instructed *not* to cite; citations are
  built from chunk metadata and a deterministic strip removes any the small model
  leaks anyway (D-9, D-12).
- **Grounding guard** — answers only from retrieved context; returns a fixed "I
  don't know" string otherwise.

## Quickstart

### With Docker (API + Qdrant)

Ollama runs on the host so the model stays on your machine:

```bash
ollama serve &            # in another terminal
ollama pull llama3.2:3b
docker compose up --build
```

### Locally

```bash
python -m pip install -r requirements.txt
ollama pull llama3.2:3b   # optional; retrieval works without it
uvicorn app.main:app --reload
```

Then open <http://localhost:8000> for a minimal demo UI, or use the API:

```bash
# Ingest the bundled roadmap PDF
curl -X POST localhost:8000/ingest/path \
  -H 'Content-Type: application/json' \
  -d '{"path": "data/documents/Shashwat_AI_ML_Project_Roadmap.pdf"}'

# Ask a question
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the ingestion pipeline order for DocuMind?"}'
```

```json
{
  "answer": "PDF/text loader → chunking → embeddings → vector store.",
  "refused": false,
  "citations": [{"source": "Shashwat_AI_ML_Project_Roadmap.pdf", "page": 1}],
  "latency_ms": {"retrieval": 544.0, "generation": 690.0}
}
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + the active configuration |
| POST | `/ingest` | upload a PDF (multipart `file`) and index it |
| POST | `/ingest/path` | index a PDF already on the server |
| POST | `/query` | retrieve and (optionally) generate a grounded answer |
| POST | `/feedback` | log a thumbs up/down — seed for a future eval set |

Interactive docs at `/docs` (FastAPI/OpenAPI).

## Configuration

All settings live in [`app/config.py`](app/config.py) with the decision-log entry
that justifies each default; override any via environment variables (prefix
`DOCUMIND_`) or a `.env` file — see [`.env.example`](.env.example).

## Repository layout

```
app/            FastAPI service (the locked pipeline)
  config.py       all hyperparameters, each annotated with its decision
  ingestion/      load, chunk + enrich, embed
  retrieval/      Qdrant store, dense search, rerank, diversify
  generation/     Prompt B, Ollama, citation strip
  main.py         HTTP endpoints
eval/           source-aware evaluation harness (python -m eval)
  golden_set.py   63 hand-grounded questions over the 4-document corpus
  harness.py      loads every PDF, runs the real app pipeline, scores it
scripts/        fetch_corpus.py (corpus), check_eval.py (CI regression gate)
tests/          fast, network-free unit tests for the pipeline invariants
notebooks/      rag_1.ipynb — the experiments that selected the config
experiment/     measured results (CSV/JSON) behind every decision
experiments.md  the experiment log
decision.md     what was decided, why, and what was rejected
.github/        CI: unit tests + the eval regression gate on every push
```

## Evaluation

The corpus is the roadmap PDF (tracked) plus three public arXiv papers (fetched,
not committed). Download them and run the harness:

```bash
python scripts/fetch_corpus.py   # -> data/documents/*.pdf
python -m eval                   # Recall@5, MRR, coverage over the golden set
```

The harness indexes every PDF through the exact `app` pipeline, so its numbers
can never drift from what the service actually does, and it keys ground truth on
`(source, page)` so a multi-document corpus is scored correctly.

## Development & tests

```bash
pip install -r requirements-dev.txt
pytest                      # fast, network-free unit tests (no model downloads)
python scripts/check_eval.py   # the regression gate: eval the golden set, fail on a drop
```

CI (`.github/workflows/ci.yml`) runs both on every push and PR:

- **Unit tests** cover the pipeline invariants — metadata enrichment, BM25, the
  query-aware fusion maths (D-17), page diversification, and the citation strip.
- **Eval regression gate** fetches the corpus and fails the build if Page
  Recall@5, MRR or coverage falls below the thresholds the shipped config has
  earned — the roadmap's "regression-tested quality" step.

## Status & roadmap

The retrieval/generation pipeline is **locked**, served, evaluated on a
multi-document corpus, and guarded by CI. Logged next steps:

- ✅ Closed 2 of the 3 residual misses with query-aware fusion (D-17). The 1
  remaining (roadmap *Ragas*) is a genuinely ambiguous grounding — the query
  points at the golden-set page, the full metric list is on another — left
  standing rather than relabelled.
- ✅ Unit tests + CI with an eval regression gate (D-18).
- Add an answer-term **accuracy** metric (not just coverage) to catch wrong-list
  answers.
- Wire **Langfuse** tracing for per-stage latency/cost.
- Add **Ragas** context-precision/recall in CI against the golden set.

## Tech stack

FastAPI · Qdrant · LangChain · sentence-transformers (all-MiniLM-L6-v2) ·
cross-encoder (ms-marco-MiniLM-L-6-v2) · Ollama (llama3.2:3b) · Docker
