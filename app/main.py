"""DocuMind FastAPI service.

Endpoints (the three from the roadmap plus health):
    GET  /health            liveness + what's wired up
    POST /ingest            load + chunk + index a PDF (upload, or a server path)
    POST /query             retrieve (+ optionally generate a grounded answer)
    POST /feedback          record a thumbs up/down on an answer

The pipeline stages are imported from the ``app`` package; this module only does
HTTP wiring, request/response schemas, and the feedback log.
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app import __version__
from app.config import settings
from app.generation.generator import generator
from app.ingestion.chunker import split_documents
from app.ingestion.loader import load_pdf
from app.retrieval.retriever import collection_size, index_chunks, retrieve

app = FastAPI(
    title="DocuMind",
    version=__version__,
    description="Document question-answering RAG service. "
    "Retrieval + generation config is the one selected in decision.md.",
)

_FEEDBACK_LOG = Path("experiment/feedback_log.jsonl")


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class IngestPathRequest(BaseModel):
    path: str = Field(..., description="Server-side path to a PDF to ingest.")


class IngestResponse(BaseModel):
    filename: str
    pages: int
    chunks_indexed: int
    collection_size: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    generate: bool = Field(True, description="Also generate a grounded answer.")


class SourceChunk(BaseModel):
    page: int | None
    source: str
    retrieval_score: float
    reranker_score: float | None
    preview: str


class Citation(BaseModel):
    source: str
    page: int | None


class QueryResponse(BaseModel):
    question: str
    answer: str | None
    refused: bool | None
    citations: list[Citation]
    sources: list[SourceChunk]
    latency_ms: dict[str, float]


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    helpful: bool
    note: str | None = None


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "indexed_chunks": collection_size(),
        "generation_available": settings.generation_enabled and generator.available,
        "config": {
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "chunk": f"{settings.chunk_size}/{settings.chunk_overlap}",
            "enrich_metadata": settings.enrich_metadata,
            "retrieval_top_k": settings.retrieval_top_k,
            "reranker_top_k": settings.reranker_top_k,
        },
    }


def _ingest(path: Path) -> IngestResponse:
    pages = load_pdf(path)
    chunks = split_documents(pages)
    n = index_chunks(chunks)
    return IngestResponse(
        filename=path.name,
        pages=len(pages),
        chunks_indexed=n,
        collection_size=collection_size(),
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile | None = File(default=None)) -> IngestResponse:
    """Ingest an uploaded PDF (multipart ``file``)."""
    if file is None:
        raise HTTPException(400, "Upload a PDF as multipart 'file', or use /ingest/path.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    dest = Path(settings.data_dir) / file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    return _ingest(dest)


@app.post("/ingest/path", response_model=IngestResponse)
def ingest_path(req: IngestPathRequest) -> IngestResponse:
    """Ingest a PDF already present on the server (e.g. the bundled roadmap)."""
    path = Path(req.path)
    if not path.exists():
        raise HTTPException(404, f"No such file: {path}")
    return _ingest(path)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if collection_size() == 0:
        raise HTTPException(409, "Nothing indexed yet. POST a PDF to /ingest first.")

    t0 = time.perf_counter()
    chunks = retrieve(req.question)
    t_retrieve = (time.perf_counter() - t0) * 1000

    sources = [
        SourceChunk(
            page=c.page,
            source=str(c.document.metadata.get("source", "document")).replace("\\", "/").split("/")[-1],
            retrieval_score=round(c.retrieval_score, 4),
            reranker_score=round(c.reranker_score, 4) if c.reranker_score is not None else None,
            preview=c.document.page_content[:200],
        )
        for c in chunks
    ]

    answer = refused = None
    citations: list[Citation] = []
    t_generate = 0.0
    if req.generate and settings.generation_enabled:
        t1 = time.perf_counter()
        result = generator.generate(req.question, chunks)
        t_generate = (time.perf_counter() - t1) * 1000
        answer, refused = result.text, result.refused
        citations = [Citation(source=c.source, page=c.page) for c in result.citations]

    return QueryResponse(
        question=req.question,
        answer=answer,
        refused=refused,
        citations=citations,
        sources=sources,
        latency_ms={"retrieval": round(t_retrieve, 1), "generation": round(t_generate, 1)},
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    """Append a feedback record to a JSONL log (seed for an eval set)."""
    _FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(), **req.model_dump()}
    with _FEEDBACK_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return {"status": "recorded"}


@app.get("/", response_class=HTMLResponse)
def demo() -> str:
    return _DEMO_HTML


_DEMO_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DocuMind</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto;
         padding: 0 1rem; line-height: 1.5; }
  h1 { margin-bottom: .2rem; } .sub { color: #888; margin-top: 0; }
  textarea, button { font: inherit; } textarea { width: 100%; padding: .6rem; }
  button { padding: .6rem 1.2rem; margin-top: .5rem; cursor: pointer; }
  #answer { white-space: pre-wrap; background: rgba(127,127,127,.12);
            padding: 1rem; border-radius: 8px; margin-top: 1rem; }
  .cite { color: #888; font-size: .85rem; margin-top: .5rem; }
  .lat { color: #888; font-size: .8rem; }
</style></head><body>
<h1>DocuMind</h1>
<p class="sub">Ask a question about the ingested document.</p>
<textarea id="q" rows="3" placeholder="e.g. What is the ingestion pipeline order?"></textarea>
<button onclick="ask()">Ask</button>
<div id="answer" hidden></div>
<script>
async function ask() {
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  const box = document.getElementById('answer');
  box.hidden = false; box.textContent = 'Thinking…';
  try {
    const r = await fetch('/query', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question:q, generate:true})});
    if (!r.ok) { box.textContent = 'Error: ' + (await r.text()); return; }
    const d = await r.json();
    const pages = d.citations.map(c => 'p.' + c.page).join(', ');
    box.innerHTML = '';
    const a = document.createElement('div'); a.textContent = d.answer || '(retrieval only)';
    box.appendChild(a);
    if (pages) { const c = document.createElement('div'); c.className='cite';
      c.textContent = 'Sources: ' + pages; box.appendChild(c); }
    const l = document.createElement('div'); l.className='lat';
    l.textContent = `retrieval ${d.latency_ms.retrieval} ms · generation ${d.latency_ms.generation} ms`;
    box.appendChild(l);
  } catch (e) { box.textContent = 'Error: ' + e; }
}
</script></body></html>"""
