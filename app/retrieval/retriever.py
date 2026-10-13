"""The retrieval stage: dense candidates -> cross-encoder rerank -> diversify.

This is the selected retrieval config (D-4, D-5, D-11): a wide dense pool of
``retrieval_top_k`` candidates, re-ranked by a cross-encoder, then capped to one
chunk per page and truncated to ``reranker_top_k`` final chunks. Dense-only;
BM25 hybrid lost on this corpus (D-11) and is not wired in here.

The vector store is a module-level singleton so the heavy client/collection is
built once. It connects to a Qdrant *server* when ``qdrant_url`` is set
(docker-compose), otherwise a local on-disk store.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import CrossEncoder

from app.config import settings
from app.ingestion.embedder import get_embeddings
from app.ingestion.loader import page_number


@dataclass
class ScoredChunk:
    document: Document
    retrieval_score: float
    reranker_score: float | None = None

    @property
    def page(self) -> int | None:
        return page_number(self.document)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    model = CrossEncoder(settings.reranker_model)
    model.predict([("warm", "up")])  # pay first-call cost at startup, not per query
    return model


@lru_cache(maxsize=1)
def _get_client() -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    return QdrantClient(path=settings.qdrant_path)


@lru_cache(maxsize=1)
def get_store() -> QdrantVectorStore:
    """Vector store singleton; creates the collection on first use."""
    client = _get_client()
    embeddings = get_embeddings()
    name = settings.qdrant_collection
    if not client.collection_exists(name):
        dim = len(embeddings.embed_query("probe"))
        client.create_collection(
            name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
    return QdrantVectorStore(client=client, collection_name=name, embedding=embeddings)


def index_chunks(chunks: list[Document]) -> int:
    """Add chunks to the collection; returns the number indexed."""
    get_store().add_documents(chunks)
    return len(chunks)


def collection_size() -> int:
    client = _get_client()
    if not client.collection_exists(settings.qdrant_collection):
        return 0
    return client.count(settings.qdrant_collection).count


def _diversify(scored: list[ScoredChunk]) -> list[ScoredChunk]:
    """Cap to max_chunks_per_page, truncate to reranker_top_k final chunks."""
    out, counts = [], {}
    cap = settings.max_chunks_per_page
    for r in scored:
        key = r.document.metadata.get("page", "x")
        if cap is not None and counts.get(key, 0) >= cap:
            continue
        out.append(r)
        counts[key] = counts.get(key, 0) + 1
        if len(out) == settings.reranker_top_k:
            break
    return out


def retrieve(query: str) -> list[ScoredChunk]:
    """Full retrieval stage for one query -> final ranked chunks."""
    store = get_store()
    raw = store.similarity_search_with_score(query, k=settings.retrieval_top_k)
    candidates = [ScoredChunk(document=d, retrieval_score=float(s)) for d, s in raw]
    if not candidates:
        return []

    if settings.reranker_enabled:
        scores = get_reranker().predict(
            [(query, c.document.page_content) for c in candidates]
        )
        for c, s in zip(candidates, scores):
            c.reranker_score = float(s)
        candidates.sort(key=lambda c: c.reranker_score, reverse=True)

    return _diversify(candidates)
