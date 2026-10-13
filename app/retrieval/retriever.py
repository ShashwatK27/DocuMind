"""The retrieval stage: candidates -> cross-encoder rerank -> fuse -> diversify.

This is the selected retrieval config (D-4, D-5, D-11, D-15, D-17): a wide
candidate pool (dense top-k, optionally augmented with BM25's top lexical
matches), re-ranked by a cross-encoder, optionally blended with the lexical
signal (query-aware fusion), then capped to one chunk per page and truncated to
``reranker_top_k`` final chunks.

The vector store is a module-level singleton so the heavy client/collection is
built once. It connects to a Qdrant *server* when ``qdrant_url`` is set
(docker-compose), otherwise a local on-disk store.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.config import settings
from app.ingestion.embedder import get_embeddings
from app.ingestion.loader import page_number


@dataclass
class ScoredChunk:
    document: Document
    retrieval_score: float
    reranker_score: float | None = None
    bm25_score: float | None = None

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


def _build_store(client: QdrantClient) -> QdrantVectorStore:
    """Wrap a client as a vector store, creating the collection if absent."""
    embeddings = get_embeddings()
    name = settings.qdrant_collection
    if not client.collection_exists(name):
        dim = len(embeddings.embed_query("probe"))
        client.create_collection(
            name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
    return QdrantVectorStore(client=client, collection_name=name, embedding=embeddings)


@lru_cache(maxsize=1)
def get_store() -> QdrantVectorStore:
    """Vector store singleton; creates the collection on first use."""
    return _build_store(_get_client())


def build_memory_store(chunks: list[Document]) -> QdrantVectorStore:
    """A throwaway in-memory store holding ``chunks`` — for eval and tests.

    Uses the same embedder, collection config and distance as the real store,
    so it exercises the production pipeline without touching disk (and so the
    evaluation can never drift from what the service actually does).
    """
    store = _build_store(QdrantClient(location=":memory:"))
    store.add_documents(chunks)
    return store


# The BM25 side of hybrid retrieval needs the chunk texts, which the vector
# store does not return at query time. We keep a process-wide corpus + BM25
# index, rebuilt by scrolling Qdrant (so it survives restarts) and invalidated
# whenever new chunks are ingested.
_corpus: dict[str, object] = {}


def _load_corpus_from_store() -> list[Document]:
    client = _get_client()
    name = settings.qdrant_collection
    if not client.collection_exists(name):
        return []
    docs, offset = [], None
    while True:
        points, offset = client.scroll(
            name, with_payload=True, with_vectors=False, limit=256, offset=offset
        )
        for p in points:
            payload = p.payload or {}
            docs.append(Document(
                page_content=payload.get("page_content", ""),
                metadata=payload.get("metadata", {}),
            ))
        if offset is None:
            break
    return docs


def get_corpus() -> tuple[list[Document], BM25Okapi | None]:
    """The indexed chunks and their BM25 index (built lazily, then cached)."""
    if "chunks" not in _corpus:
        chunks = _load_corpus_from_store()
        _corpus["chunks"] = chunks
        _corpus["bm25"] = build_bm25(chunks) if chunks else None
    return _corpus["chunks"], _corpus["bm25"]  # type: ignore[return-value]


def index_chunks(chunks: list[Document]) -> int:
    """Add chunks to the collection; returns the number indexed."""
    get_store().add_documents(chunks)
    _corpus.clear()  # force a BM25 rebuild on the next hybrid query
    return len(chunks)


def collection_size() -> int:
    client = _get_client()
    if not client.collection_exists(settings.qdrant_collection):
        return 0
    return client.count(settings.qdrant_collection).count


def _tok(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25(chunks: list[Document]) -> BM25Okapi:
    """A BM25 index over the chunk texts, for the hybrid retrieval path."""
    return BM25Okapi([_tok(c.page_content) for c in chunks])


def _chunk_key(doc: Document) -> tuple:
    # Identity across the dense and BM25 rankings; source-aware for multi-doc.
    return (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:80])


def _dense_candidates(store: QdrantVectorStore, query: str, k: int) -> list[ScoredChunk]:
    raw = store.similarity_search_with_score(query, k=k)
    return [ScoredChunk(document=d, retrieval_score=float(s)) for d, s in raw]


def _hybrid_candidates(store: QdrantVectorStore, query: str, k: int,
                       chunks: list[Document], bm25: BM25Okapi) -> list[ScoredChunk]:
    """Augment the dense candidate pool with BM25's top lexical matches (D-15).

    Rather than Reciprocal-Rank Fusion (which re-truncates to ``k`` and can evict
    a correct dense chunk before the reranker sees it), take the *union* of the
    dense top-k and the BM25 top-k. The downstream cross-encoder then re-scores
    the whole pool, so BM25 can only *add* exact-match candidates (rare tokens it
    nails but dense similarity dilutes), never remove a dense one.
    """
    bm_scores = bm25.get_scores(_tok(query))
    key_to_bm = {_chunk_key(chunks[i]): float(bm_scores[i]) for i in range(len(chunks))}

    dense = store.similarity_search_with_score(query, k=k)
    candidates = [
        ScoredChunk(document=d, retrieval_score=float(s),
                    bm25_score=key_to_bm.get(_chunk_key(d), 0.0))
        for d, s in dense
    ]
    seen = {_chunk_key(c.document) for c in candidates}

    bm_order = sorted(range(len(chunks)), key=lambda i: bm_scores[i], reverse=True)[:k]
    for i in bm_order:
        key = _chunk_key(chunks[i])
        if key not in seen:
            seen.add(key)
            candidates.append(ScoredChunk(document=chunks[i], retrieval_score=0.0,
                                          bm25_score=float(bm_scores[i])))
    return candidates


def _zscore(values: list[float]) -> list[float]:
    """Standardize a list of scores; all-equal input maps to zeros."""
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    if var <= 1e-12:
        return [0.0] * n
    sd = var ** 0.5
    return [(v - mean) / sd for v in values]


def _fuse(candidates: list[ScoredChunk], alpha: float) -> None:
    """Blend the cross-encoder score with the lexical (BM25) signal in place (D-17).

    The reranker is a strong but general relevance model; on a multi-document
    corpus it sometimes ranks the exact-match chunk just below the top-5 (the
    optimizer page, the FAISS/HNSW page). BM25 nails those rare tokens. Fusing
    the two *query-normalized* signals lets a chunk both agree on rise, without
    letting either dominate. ``reranker_score`` is overwritten with the fused
    value so the existing sort/diversify path uses it unchanged.
    """
    rr_z = _zscore([c.reranker_score for c in candidates])
    bm_z = _zscore([c.bm25_score or 0.0 for c in candidates])
    for c, rz, bz in zip(candidates, rr_z, bm_z):
        c.reranker_score = rz + alpha * bz


def _diversify(scored: list[ScoredChunk]) -> list[ScoredChunk]:
    """Cap to max_chunks_per_page, truncate to reranker_top_k final chunks."""
    out, counts = [], {}
    cap = settings.max_chunks_per_page
    for r in scored:
        # Key on (source, page) so the same page number in two different
        # documents is not treated as one page (multi-document correctness).
        key = (r.document.metadata.get("source"), r.document.metadata.get("page", "x"))
        if cap is not None and counts.get(key, 0) >= cap:
            continue
        out.append(r)
        counts[key] = counts.get(key, 0) + 1
        if len(out) == settings.reranker_top_k:
            break
    return out


def retrieve(query: str, store: QdrantVectorStore | None = None, *,
             chunks: list[Document] | None = None, bm25: BM25Okapi | None = None,
             use_hybrid: bool | None = None) -> list[ScoredChunk]:
    """Full retrieval stage for one query -> final ranked chunks.

    ``store`` defaults to the service singleton; the eval harness and tests pass
    an in-memory store so they run this exact code path against their own index.
    Hybrid (dense pool augmented with BM25) runs when ``use_hybrid`` is True
    (defaulting to ``settings.hybrid_enabled``) and a ``chunks`` corpus + ``bm25``
    index are supplied; otherwise retrieval is dense-only. ``fusion_alpha`` then
    blends the lexical signal into the final ranking (D-17).
    """
    store = store or get_store()
    use_hybrid = settings.hybrid_enabled if use_hybrid is None else use_hybrid
    k = settings.retrieval_top_k

    if use_hybrid:
        # Eval/tests pass the corpus explicitly; the service loads it from Qdrant.
        if chunks is None or bm25 is None:
            chunks, bm25 = get_corpus()
        if chunks and bm25 is not None:
            candidates = _hybrid_candidates(store, query, k, chunks, bm25)
        else:
            candidates = _dense_candidates(store, query, k)
    else:
        candidates = _dense_candidates(store, query, k)
    if not candidates:
        return []

    if settings.reranker_enabled:
        scores = get_reranker().predict(
            [(query, c.document.page_content) for c in candidates]
        )
        for c, s in zip(candidates, scores):
            c.reranker_score = float(s)
        # Query-aware fusion: nudge the ranking toward chunks the lexical index
        # also likes, recovering exact-match pages the reranker buries (D-17).
        if settings.fusion_alpha > 0 and any(c.bm25_score is not None for c in candidates):
            _fuse(candidates, settings.fusion_alpha)
        candidates.sort(key=lambda c: c.reranker_score, reverse=True)

    return _diversify(candidates)
