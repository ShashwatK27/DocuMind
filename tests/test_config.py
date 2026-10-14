"""The shipped configuration is a set of measured decisions; guard the defaults
so a stray edit can't silently change the served pipeline."""
from app.config import settings


def test_chunking_defaults():
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 150
    assert settings.enrich_metadata is True


def test_retrieval_defaults():
    assert settings.retrieval_top_k == 20
    assert settings.reranker_enabled is True
    assert settings.reranker_top_k == 5
    assert settings.max_chunks_per_page == 1


def test_hybrid_and_fusion_defaults():
    assert settings.hybrid_enabled is True
    # D-17: the swept knee. 0.0 would silently revert to pure-reranker ranking.
    assert settings.fusion_alpha == 0.15
