"""Chunking + metadata enrichment invariants (D-3, D-11)."""
import pytest
from langchain_core.documents import Document

from app.config import settings
from app.ingestion.chunker import split_documents

PAGE = Document(
    page_content=(
        "INTRODUCTION\n"
        "The Transformer is a model architecture based solely on attention "
        "mechanisms, dispensing with recurrence and convolutions entirely."
    ),
    metadata={"source": "data/documents/paper.pdf", "page": 0},
)


@pytest.fixture
def enrich_on():
    old = settings.enrich_metadata
    settings.enrich_metadata = True
    yield
    settings.enrich_metadata = old


def test_enrichment_prepends_source_page_section(enrich_on):
    [chunk] = split_documents([PAGE])
    # page is 0-based in metadata; the human-facing prefix is 1-based.
    assert chunk.page_content.startswith(
        "Source: paper.pdf | Page 1 | Section: INTRODUCTION\n"
    )
    # The original body survives after the header.
    assert "attention mechanisms" in chunk.page_content
    # Metadata is preserved (source/page) for source-aware scoring.
    assert chunk.metadata["source"].endswith("paper.pdf")
    assert chunk.metadata["page"] == 0


def test_enrichment_can_be_disabled():
    old = settings.enrich_metadata
    settings.enrich_metadata = False
    try:
        [chunk] = split_documents([PAGE])
        assert not chunk.page_content.startswith("Source:")
        assert chunk.page_content.startswith("INTRODUCTION")
    finally:
        settings.enrich_metadata = old


def test_long_page_splits_into_multiple_chunks(enrich_on):
    body = ". ".join(f"sentence number {i} about retrieval" for i in range(400))
    page = Document(page_content="OVERVIEW\n" + body,
                    metadata={"source": "x.pdf", "page": 2})
    chunks = split_documents([page])
    assert len(chunks) > 1
    assert all(c.page_content.startswith("Source: x.pdf | Page 3") for c in chunks)
