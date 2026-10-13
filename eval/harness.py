"""Run the golden set through the real pipeline and report source-aware metrics.

Usage:
    python -m eval                 # evaluate every PDF in data/documents/
    python -m eval --no-csv        # print only, don't write result files

Metrics (over the answerable questions):
    Page Recall@5  did any of the final top-5 chunks come from an expected
                   (source, page)?
    MRR            reciprocal rank of the first chunk from an expected (source, page).
    Coverage       fraction of ground-truth answer terms present in the delivered
                   context (and the share of questions with coverage >= 0.5).

Unanswerable questions are reported separately (they have no expected page).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from langchain_core.documents import Document

from app.config import settings
from app.ingestion.chunker import split_documents
from app.ingestion.loader import load_pdf, page_number
from app.retrieval.retriever import build_bm25, build_memory_store, retrieve
from eval.golden_set import golden_set


def _basename(source: object) -> str:
    return Path(str(source)).name


def _answer_terms(ground_truth: str) -> set[str]:
    return {t.strip(".,;:()") for t in ground_truth.lower().split()
            if len(t.strip(".,;:()")) >= 5}


def content_coverage(contexts: list[str], ground_truth: str) -> float:
    terms = _answer_terms(ground_truth)
    if not terms:
        return 0.0
    text = " ".join(contexts).lower()
    return sum(t in text for t in terms) / len(terms)


def load_corpus() -> tuple[list[Document], dict[tuple[str, int], str]]:
    """Load every PDF in the data dir; return its chunks and a page-text index."""
    pdfs = sorted(Path(settings.data_dir).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {settings.data_dir}")
    chunks: list[Document] = []
    page_text: dict[tuple[str, int], str] = {}
    for pdf in pdfs:
        pages = load_pdf(pdf)
        chunks.extend(split_documents(pages))
        for d in pages:
            pn = page_number(d)
            if pn is not None:
                page_text[(pdf.name, pn)] = d.page_content.lower()
    return chunks, page_text


def check_grounding(page_text: dict[tuple[str, int], str]) -> list[str]:
    """Warn when a question's answer terms are largely absent from its claimed page."""
    warnings: list[str] = []
    for rec in golden_set:
        if not rec["source_pages"]:
            continue
        on_page = " ".join(
            page_text.get((rec["source"], p), "") for p in rec["source_pages"]
        )
        terms = _answer_terms(rec["ground_truth"])
        missing = [t for t in terms if t not in on_page]
        if terms and len(missing) / len(terms) > 0.5:
            warnings.append(f"  [{rec['source']} p{rec['source_pages']}] "
                            f"{rec['question']!r} missing {missing}")
    return warnings


def evaluate() -> tuple[dict, list[dict]]:
    chunks, _ = load_corpus()
    store = build_memory_store(chunks)
    bm25 = build_bm25(chunks)
    return _score(store, chunks, bm25, use_hybrid=settings.hybrid_enabled)


def _score(store, chunks, bm25, use_hybrid: bool) -> tuple[dict, list[dict]]:
    """Score the golden set with one retrieval mode against a prebuilt corpus."""
    per_q: list[dict] = []
    for rec in golden_set:
        final = retrieve(rec["question"], store=store, chunks=chunks,
                         bm25=bm25, use_hybrid=use_hybrid)
        retrieved = [(_basename(c.document.metadata.get("source")), c.page) for c in final]
        contexts = [c.document.page_content for c in final]
        expected = {(rec["source"], p) for p in rec["source_pages"]}
        answerable = bool(expected)

        recall5 = answerable and any(rp in expected for rp in retrieved[:5])
        rr = 0.0
        for rank, rp in enumerate(retrieved, 1):
            if rp in expected:
                rr = 1 / rank
                break
        cov = content_coverage(contexts, rec["ground_truth"]) if answerable else None
        per_q.append({"source": rec["source"], "question": rec["question"],
                      "type": rec["type"], "answerable": answerable,
                      "retrieved": retrieved, "recall5": recall5, "mrr": rr,
                      "coverage": cov})

    ans = [r for r in per_q if r["answerable"]]
    metrics = {
        "documents": len({_basename(c.metadata.get("source")) for c in chunks}),
        "chunks": len(chunks),
        "questions": len(golden_set),
        "answerable": len(ans),
        "page_recall@5": sum(r["recall5"] for r in ans) / len(ans),
        "mrr": sum(r["mrr"] for r in ans) / len(ans),
        "coverage": sum(r["coverage"] for r in ans) / len(ans),
        "coverage>=0.5": sum(r["coverage"] >= 0.5 for r in ans) / len(ans),
    }
    return metrics, per_q


def _print_metrics(title: str, metrics: dict) -> None:
    print(f"=== {title} ===")
    for k, v in metrics.items():
        print(f"  {k:>16}: {v:.3f}" if isinstance(v, float) else f"  {k:>16}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DocuMind evaluation harness")
    parser.add_argument("--no-csv", action="store_true", help="don't write result CSVs")
    parser.add_argument("--compare", action="store_true",
                        help="evaluate dense vs hybrid (BM25+dense RRF) side by side")
    args = parser.parse_args()

    chunks, page_text = load_corpus()
    warnings = check_grounding(page_text)
    if warnings:
        print("Grounding warnings (answer terms missing from claimed page):")
        print("\n".join(warnings), "\n")

    store = build_memory_store(chunks)
    bm25 = build_bm25(chunks)

    if not args.compare:
        metrics, per_q = _score(store, chunks, bm25, use_hybrid=settings.hybrid_enabled)
        _print_metrics("DocuMind evaluation", metrics)
        misses = [r for r in per_q if r["answerable"] and not r["recall5"]]
        if misses:
            print("\nRecall@5 misses:")
            for r in misses:
                print(f"  [{r['source']}] {r['question']!r} -> {r['retrieved'][:5]}")
        if not args.no_csv:
            _write_csv("eval", metrics, per_q)
        return

    # --- dense vs hybrid comparison (D-15) ---
    dense_m, dense_q = _score(store, chunks, bm25, use_hybrid=False)
    hybrid_m, hybrid_q = _score(store, chunks, bm25, use_hybrid=True)
    _print_metrics("Dense (baseline)", dense_m)
    print()
    _print_metrics("Hybrid (dense pool augmented with BM25)", hybrid_m)

    print("\n=== Delta (hybrid - dense) ===")
    for k in ("page_recall@5", "mrr", "coverage", "coverage>=0.5"):
        print(f"  {k:>16}: {hybrid_m[k] - dense_m[k]:+.3f}")

    dense_miss = {r["question"] for r in dense_q if r["answerable"] and not r["recall5"]}
    hybrid_miss = {r["question"] for r in hybrid_q if r["answerable"] and not r["recall5"]}
    recovered = dense_miss - hybrid_miss
    introduced = hybrid_miss - dense_miss
    print(f"\nRecall@5 misses: dense {len(dense_miss)} -> hybrid {len(hybrid_miss)}")
    for q in sorted(recovered):
        print(f"  RECOVERED by hybrid: {q!r}")
    for q in sorted(introduced):
        print(f"  NEW miss from hybrid: {q!r}")
    for q in sorted(hybrid_miss & dense_miss):
        print(f"  still missed: {q!r}")

    if not args.no_csv:
        import csv
        out = Path("experiment")
        out.mkdir(exist_ok=True)
        with (out / "hybrid_vs_dense_results.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["metric", "dense", "hybrid", "delta"])
            for k in ("page_recall@5", "mrr", "coverage", "coverage>=0.5"):
                w.writerow([k, round(dense_m[k], 4), round(hybrid_m[k], 4),
                            round(hybrid_m[k] - dense_m[k], 4)])
        print(f"\nWrote {out/'hybrid_vs_dense_results.csv'}")


def _write_csv(prefix: str, metrics: dict, per_q: list[dict]) -> None:
    import csv
    out = Path("experiment")
    out.mkdir(exist_ok=True)
    with (out / f"{prefix}_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])
    with (out / f"{prefix}_per_question.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "question", "type",
                                           "answerable", "recall5", "mrr", "coverage"])
        w.writeheader()
        for r in per_q:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"\nWrote {out/f'{prefix}_summary.csv'} and {out/f'{prefix}_per_question.csv'}")


if __name__ == "__main__":
    main()
