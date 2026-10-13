"""Chunking + metadata enrichment (the ingestion-side feature engineering).

Recursive character splitting at the selected 500/150 (D-3, D-11). When
enrichment is on, each chunk is prefixed with ``Source | Page | Section`` — the
nearest preceding heading on its page — which the experiments measured as a
coverage win at overlap 150 (D-11). The prefix is part of the embedded text, so
it also improves lexical/semantic recall of section-scoped queries.
"""
from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

# Plain separators were selected; structure-aware bullet separators were
# neutral on this corpus (D-11) and are intentionally not used here.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# A heading is a numbered/all-caps/"Title:" line — used to tag each chunk with
# the section it came from.
_HEADING = re.compile(
    r"^(PROJECT\s+\d+.*|[0-9]+\.\s+[A-Z].*|[A-Z][A-Za-z0-9 /&+\-]{2,60}:|[A-Z][A-Z /&]{3,60})$"
)


def _page_headings(text: str) -> list[tuple[int, str]]:
    """(char offset, heading) for heading-like lines on a page, in order."""
    out, pos = [], 0
    for line in text.splitlines(keepends=True):
        s = line.strip()
        if s and _HEADING.match(s):
            out.append((pos, s.rstrip(":")))
        pos += len(line)
    return out


def _heading_for(page_text: str, chunk_text: str, headings: list[tuple[int, str]]) -> str | None:
    """Nearest heading at or before where this chunk starts on its page."""
    start = page_text.find(chunk_text[:60])
    if start < 0:
        return headings[0][1] if headings else None
    chosen = None
    for off, h in headings:
        if off <= start:
            chosen = h
        else:
            break
    return chosen or (headings[0][1] if headings else None)


def split_documents(documents: list[Document]) -> list[Document]:
    """Chunk pages into retrieval units, enriching with metadata if configured."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=_SEPARATORS,
    )
    chunks = splitter.split_documents(documents)
    if not settings.enrich_metadata:
        return chunks

    page_text = {d.metadata.get("page"): d.page_content for d in documents}
    page_heads = {p: _page_headings(t) for p, t in page_text.items()}
    enriched: list[Document] = []
    for c in chunks:
        p = c.metadata.get("page")
        src = Path(str(c.metadata.get("source", "document"))).name
        heading = _heading_for(page_text.get(p, ""), c.page_content, page_heads.get(p, []))
        header = f"Source: {src} | Page {(p + 1) if isinstance(p, int) else '?'}"
        if heading:
            header += f" | Section: {heading}"
        enriched.append(
            Document(page_content=f"{header}\n{c.page_content}", metadata=dict(c.metadata))
        )
    return enriched
