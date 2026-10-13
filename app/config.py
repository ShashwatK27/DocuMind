"""Central configuration for the DocuMind service.

Every retrieval/generation value below is a *decision*, not a guess — each is
annotated with the decision-log entry that justifies it (see ``decision.md``).
Settings may be overridden from the environment or a ``.env`` file; the defaults
are the configuration the experiments selected.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCUMIND_", env_file=".env", extra="ignore"
    )

    # --- Models ------------------------------------------------------------
    # all-MiniLM-L6-v2 beat L12 on both MRR and latency (D-5). Normalized
    # embeddings + cosine distance. CPU keeps the service portable.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Chunking (D-3, D-11) ---------------------------------------------
    # 500/150 with metadata enrichment was the selected config: overlap 150
    # (not 100) because answer lists span chunk boundaries; enrichment helps
    # *only* at overlap 150, so the two were tuned jointly.
    chunk_size: int = 500
    chunk_overlap: int = 150
    enrich_metadata: bool = True

    # --- Retrieval (D-4, D-5, D-11, D-15) ---------------------------------
    # Wide dense candidate pool -> cross-encoder rerank -> page diversify.
    retrieval_top_k: int = 20
    reranker_enabled: bool = True
    reranker_top_k: int = 5          # final chunks handed to the LLM
    max_chunks_per_page: int = 1     # page diversification cap (D-5)

    # Hybrid retrieval: augment the dense pool with BM25's top lexical matches,
    # then let the reranker decide. Measured as a clean win on the 4-document
    # corpus (D-15): Recall@5 +0.017, MRR +0.038, coverage +0.032, no regressions.
    hybrid_enabled: bool = True

    # Query-aware fusion (D-17): after reranking, blend the cross-encoder score
    # with the query-normalized BM25 score (reranker_z + alpha * bm25_z) so an
    # exact-match chunk both signals agree on rises into the top-5. alpha swept
    # on the 4-doc corpus: 0.15 is the knee (Recall@5 0.949->0.983, MRR +0.007,
    # no regressions); >=0.5 starts eroding MRR, 1.0 adds new misses. 0.0
    # disables (pure reranker). Only active with hybrid.
    fusion_alpha: float = 0.15

    # --- Generation (D-9, D-12) -------------------------------------------
    # Ollama llama3.2:3b, temp 0, answer-preferring Prompt B, with a
    # deterministic citation-stripping backstop (Python owns citations).
    ollama_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"
    generation_enabled: bool = True
    unknown_answer: str = "I don't know based on the provided documents."

    # --- Vector store ------------------------------------------------------
    # Prefer a Qdrant server (docker-compose) via URL; else a local path.
    # A server sidesteps the OneDrive sqlite-lock problem seen in dev.
    qdrant_url: str | None = None
    qdrant_path: str = "qdrant_storage"
    qdrant_collection: str = "documind"

    # --- Service -----------------------------------------------------------
    data_dir: str = "data/documents"


settings = Settings()
