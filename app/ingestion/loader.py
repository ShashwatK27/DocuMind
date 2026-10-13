"""PDF loading.

One page -> one LangChain ``Document`` with page metadata, via ``PyPDFLoader``
(the loader the experiments used). Pages are 0-based in PyPDF metadata; the rest
of the app treats pages as 1-based for humans (see ``page_number``).
"""
from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(path: str | Path) -> list[Document]:
    """Load a single PDF into per-page documents."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    return PyPDFLoader(str(path)).load()


def page_number(doc: Document) -> int | None:
    """1-based page number for a document, or None if unknown."""
    p = doc.metadata.get("page")
    return p + 1 if isinstance(p, int) else None
