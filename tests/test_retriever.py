"""Retrieval-stage pure logic: BM25, z-scoring, query-aware fusion (D-17),
page diversification (D-5). No model downloads or vector store needed."""
import pytest
from langchain_core.documents import Document

from app.config import settings
from app.retrieval.retriever import (
    ScoredChunk, _chunk_key, _diversify, _fuse, _tok, _zscore, build_bm25,
)


def _doc(source, page, text):
    return Document(page_content=text, metadata={"source": source, "page": page})


def _sc(source, page, text, rr, bm):
    return ScoredChunk(document=_doc(source, page, text), retrieval_score=0.0,
                       reranker_score=rr, bm25_score=bm)


# --- tokenization & BM25 ---------------------------------------------------

def test_tok_lowercases_and_splits_alnum():
    assert _tok("FAISS, HNSW-index v2!") == ["faiss", "hnsw", "index", "v2"]


def test_bm25_ranks_the_lexically_matching_chunk_first():
    chunks = [
        _doc("a.pdf", 1, "the adam optimizer varies the learning rate"),
        _doc("a.pdf", 2, "the encoder is a stack of six identical layers"),
        _doc("a.pdf", 3, "residual dropout and label smoothing regularization"),
    ]
    bm25 = build_bm25(chunks)
    scores = bm25.get_scores(_tok("which optimizer trains the model"))
    assert scores[0] == max(scores)  # the "optimizer" chunk wins


# --- z-scoring -------------------------------------------------------------

def test_zscore_all_equal_is_zeros():
    assert _zscore([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0]


def test_zscore_empty():
    assert _zscore([]) == []


def test_zscore_is_standardized():
    z = _zscore([1.0, 2.0, 3.0])
    assert abs(sum(z)) < 1e-9          # mean ~ 0
    assert z[0] < z[1] < z[2]          # order preserved


# --- query-aware fusion (D-17) ---------------------------------------------

def test_fuse_overwrites_score_with_rerankerz_plus_alpha_bm25z():
    rr, bm, alpha = [2.0, 1.0, 0.0], [0.0, 0.0, 3.0], 0.5
    cands = [_sc("a.pdf", i, "t", r, b) for i, (r, b) in enumerate(zip(rr, bm))]
    _fuse(cands, alpha=alpha)
    expected = [rz + alpha * bz for rz, bz in zip(_zscore(rr), _zscore(bm))]
    assert [c.reranker_score for c in cands] == pytest.approx(expected)


def test_fuse_promotion_scales_with_alpha():
    # Chunk #2 trails on the reranker but dominates BM25. A light touch leaves it
    # last; enough lexical weight promotes it to first (the D-17 sweep in miniature).
    def fused_order(alpha):
        cands = [_sc("a.pdf", i, "t", r, b)
                 for i, (r, b) in enumerate([(2.0, 0.0), (1.0, 0.0), (0.0, 3.0)])]
        _fuse(cands, alpha=alpha)
        return sorted(range(3), key=lambda i: cands[i].reranker_score, reverse=True)

    assert fused_order(0.0)[-1] == 2   # pure reranker: lexical chunk is last
    assert fused_order(2.0)[0] == 2    # heavy lexical weight: it leads


def test_fuse_with_zero_bm25_variance_is_pure_reranker_order():
    a = _sc("a.pdf", 1, "x", rr=2.0, bm=5.0)
    b = _sc("a.pdf", 2, "y", rr=1.0, bm=5.0)  # equal BM25 -> bm_z all zeros
    _fuse([a, b], alpha=0.5)
    assert a.reranker_score > b.reranker_score


# --- page diversification (D-5) --------------------------------------------

def test_diversify_caps_one_chunk_per_source_page():
    scored = [
        _sc("a.pdf", 1, "first on page 1", rr=5.0, bm=0.0),
        _sc("a.pdf", 1, "second on page 1", rr=4.0, bm=0.0),   # dropped (cap 1)
        _sc("a.pdf", 2, "page 2", rr=3.0, bm=0.0),
        _sc("b.pdf", 1, "other doc page 1", rr=2.0, bm=0.0),   # not merged w/ a.pdf p1
    ]
    out = _diversify(scored)
    keys = [(c.document.metadata["source"], c.document.metadata["page"]) for c in out]
    assert keys == [("a.pdf", 1), ("a.pdf", 2), ("b.pdf", 1)]


def test_diversify_truncates_to_reranker_top_k():
    scored = [_sc("a.pdf", p, f"page {p}", rr=float(10 - p), bm=0.0) for p in range(10)]
    out = _diversify(scored)
    assert len(out) == settings.reranker_top_k


# --- chunk identity --------------------------------------------------------

def test_chunk_key_is_source_page_and_prefix():
    d = _doc("a.pdf", 3, "x" * 200)
    assert _chunk_key(d) == ("a.pdf", 3, "x" * 80)
