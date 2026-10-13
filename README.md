# DocuMind — a document question-answering RAG service

Ask questions about a PDF and get a grounded, cited answer — or an honest "I
don't know" when the document doesn't support one. DocuMind is a retrieval-
augmented generation (RAG) service whose every hyperparameter was **chosen from
measured experiments**, not guessed. The experiment log (`experiments.md`) and
the decision log (`decision.md`) are first-class parts of this repo.

## Results (selected configuration, 43-question evaluation set)

| Metric | Value | What it means |
|---|---:|---|
| Page Recall@5 | **1.000** | the right page is always in the top-5 retrieved chunks |
| MRR | **0.944** | the right page is almost always ranked first |
| Content coverage | **0.759** | fraction of ground-truth answer terms delivered in context |
| Citation leaks | **0 / 43** | the LLM never leaks source labels into prose |
| Refusals on answerable Q | **0 / 39** | it answers everything it can |
| Unanswerable refused | **4 / 4** | it refuses everything it can't ground |

Full ablations (chunk size, top-k, reranker on/off, metadata enrichment, hybrid
search, prompt A/B) are in [`experiments.md`](experiments.md); the rationale and
rejected alternatives for each choice are in [`decision.md`](decision.md).

## Pipeline

```
PDF ─► load (PyPDF) ─► chunk 500/150 + metadata enrichment ─► embed (MiniLM-L6-v2, cosine)
                                                                      │
query ─► embed ─► dense top-20 ─► cross-encoder rerank ─► 1 chunk/page, top-5 ─► LLM (Prompt B)
                                                                      │
                                                        grounded answer + Python-owned citations
```

Key design decisions:

- **Metadata enrichment** — each chunk is prefixed with `Source | Page | Section`;
  measured to raise content coverage at overlap 150 (D-11).
- **Cross-encoder reranking** over a wide dense pool — lifts MRR at every top-k (D-4).
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
notebooks/      rag_1.ipynb — the experiments that selected the config
experiment/     measured results (CSV/JSON) behind every decision
experiments.md  the experiment log
decision.md     what was decided, why, and what was rejected
```

## Status & roadmap

The retrieval/generation pipeline is **locked** and served. Logged next steps:

- Evaluate on a **multi-document corpus** (likely revives hybrid search, which
  lost on this single-doc set — D-11).
- Wire **Langfuse** tracing for per-stage latency/cost.
- Add **Ragas** context-precision/recall in CI against the golden set.
- Answer-term **accuracy** metric (not just coverage) to catch wrong-list answers.

## Tech stack

FastAPI · Qdrant · LangChain · sentence-transformers (all-MiniLM-L6-v2) ·
cross-encoder (ms-marco-MiniLM-L-6-v2) · Ollama (llama3.2:3b) · Docker
