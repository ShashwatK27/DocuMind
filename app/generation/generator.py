"""Grounded generation over retrieved context.

Three invariants, all from the experiments:

* **Prompt B** (D-12): answer if the context supports it even partially; reply
  with the exact UNKNOWN string only when nothing is relevant. This both stopped
  over-refusal and removed the self-contradiction in the old prompt.
* **Python owns citations** (notebook invariant): the model is told *not* to cite;
  the API returns a structured ``citations`` list built from chunk metadata.
* **Citation strip** (D-9): a deterministic backstop removes any source/page
  labels the small model leaks despite the instruction. The exact refusal string
  is preserved untouched.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_ollama import ChatOllama

from app.config import settings
from app.ingestion.loader import page_number
from app.retrieval.retriever import ScoredChunk

_PROMPT_B = (
    "You are answering a question using only the context below.\n\n"
    "Instructions:\n"
    "- Answer using ONLY information present in the context. Do not use outside knowledge.\n"
    "- If the context contains the answer even partially, extract and state it. "
    "Prefer answering over refusing.\n"
    '- Only reply exactly "{unknown}" if the context genuinely contains nothing relevant.\n'
    "- Do NOT write citations, source numbers, page numbers or labels; "
    "the app adds them separately.\n"
    "- Be direct and concise.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

# Leaked labels: "[SOURCE 2 | Page 3]", "Page 4", "[2]", "SOURCE 1".
_LEAK = re.compile(
    r"\[?\s*source\s*\d+(\s*\|\s*page\s*\d+)?\s*\]?|\bpage\s*\d+\b|\[\d+\]",
    re.IGNORECASE,
)
# Dangling connective left behind after a label is stripped ("according to ,").
_DANGLING = re.compile(r"\baccording to\s*[,:]?\s*", re.IGNORECASE)


@dataclass
class Citation:
    source: str
    page: int | None


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    refused: bool


def strip_citation_markers(text: str) -> str:
    """Remove any source/page labels the model emitted, and tidy up."""
    cleaned = _LEAK.sub("", text)
    cleaned = _DANGLING.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return cleaned.strip()


def _format_context(chunks: list[ScoredChunk]) -> str:
    return "\n\n".join(
        f"[SOURCE {i} | Page {c.page}]\n{c.document.page_content}"
        for i, c in enumerate(chunks, 1)
    )


def _citations(chunks: list[ScoredChunk]) -> list[Citation]:
    seen, out = set(), []
    for c in chunks:
        src = c.document.metadata.get("source", "document")
        key = (src, page_number(c.document))
        if key in seen:
            continue
        seen.add(key)
        out.append(Citation(source=str(src).split("\\")[-1].split("/")[-1], page=key[1]))
    return out


class Generator:
    """Lazily-constructed LLM client with a connection probe (D-1)."""

    def __init__(self) -> None:
        self._llm: ChatOllama | None = None
        self._probed = False

    def _ensure_llm(self) -> ChatOllama | None:
        if self._probed:
            return self._llm
        self._probed = True
        try:
            candidate = ChatOllama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=0,
            )
            candidate.invoke("ping")
            self._llm = candidate
        except Exception:
            self._llm = None  # degrade to retrieval-only
        return self._llm

    @property
    def available(self) -> bool:
        return self._ensure_llm() is not None

    def generate(self, question: str, chunks: list[ScoredChunk]) -> Answer:
        citations = _citations(chunks)
        unknown = settings.unknown_answer
        if not chunks:
            return Answer(text=unknown, citations=[], refused=True)

        llm = self._ensure_llm()
        if llm is None:
            raise RuntimeError(
                "Generation requested but Ollama is unreachable. Start Ollama "
                f"({settings.ollama_model}) or call /query with generate=false."
            )

        prompt = _PROMPT_B.format(
            unknown=unknown, context=_format_context(chunks), question=question
        )
        raw = llm.invoke(prompt).content.strip()
        refused = unknown.lower()[:20] in raw.lower()
        text = raw if refused else strip_citation_markers(raw)
        return Answer(text=text, citations=[] if refused else citations, refused=refused)


generator = Generator()
