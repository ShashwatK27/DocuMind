"""Embedding model, built once and shared.

all-MiniLM-L6-v2, normalized, on CPU (D-5). Loading the model is expensive, so
it is cached process-wide via ``functools.lru_cache``.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )
