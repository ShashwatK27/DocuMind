"""Generation-stage invariants that don't need a live LLM: the citation strip
(D-9), the refusal path, and Python-owned citations."""
from langchain_core.documents import Document

from app.config import settings
from app.generation.generator import (
    Generator, strip_citation_markers, _citations,
)
from app.retrieval.retriever import ScoredChunk


def _sc(source, page, text):
    return ScoredChunk(
        document=Document(page_content=text, metadata={"source": source, "page": page}),
        retrieval_score=1.0,
    )


# --- citation strip (D-9) --------------------------------------------------

def test_strip_removes_bracketed_source_page_label():
    out = strip_citation_markers("The model uses Adam [SOURCE 2 | Page 7] for training.")
    assert "SOURCE" not in out and "Page" not in out
    assert out == "The model uses Adam for training."


def test_strip_removes_dangling_according_to_connective():
    out = strip_citation_markers("According to SOURCE 1 | Page 3, the encoder has six layers.")
    assert out == "the encoder has six layers."


def test_strip_leaves_clean_answer_untouched():
    clean = "The Transformer dispenses with recurrence and convolutions."
    assert strip_citation_markers(clean) == clean


def test_strip_preserves_exact_refusal_string():
    # The refusal must survive verbatim so the API can flag refused=True.
    assert strip_citation_markers(settings.unknown_answer) == settings.unknown_answer


# --- citations are built from metadata, deduped, basename-only -------------

def test_citations_dedupe_by_source_page_and_use_basename():
    chunks = [
        _sc("C:\\docs\\paper.pdf", 0, "a"),
        _sc("C:\\docs\\paper.pdf", 0, "b"),   # same (source, page) -> one citation
        _sc("C:\\docs\\paper.pdf", 1, "c"),
    ]
    cites = _citations(chunks)
    assert len(cites) == 2
    assert all(c.source == "paper.pdf" for c in cites)


# --- refusal path needs no LLM ---------------------------------------------

def test_generate_with_no_chunks_refuses_without_calling_llm():
    ans = Generator().generate("anything?", [])
    assert ans.refused is True
    assert ans.citations == []
    assert ans.text == settings.unknown_answer
