"""Regression gate: run the golden-set evaluation and fail if retrieval quality
drops below the thresholds the shipped config has earned.

This is the roadmap's "regression-tested quality" step (page 2): CI runs it on
every push, so a change that quietly degrades retrieval breaks the build instead
of shipping. Thresholds sit a margin below the current measured numbers
(Recall@5 0.983, MRR 0.897, coverage 0.754 — see experiments.md / D-17) to allow
for library-version drift without masking a real regression.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/check_eval.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import evaluate  # noqa: E402

THRESHOLDS = {
    "page_recall@5": 0.95,
    "mrr": 0.88,
    "coverage": 0.72,
}


def main() -> int:
    metrics, _ = evaluate()
    print(f"corpus: {metrics['documents']} docs, {metrics['chunks']} chunks, "
          f"{metrics['answerable']}/{metrics['questions']} answerable\n")

    failures = []
    for name, floor in THRESHOLDS.items():
        value = metrics[name]
        ok = value >= floor
        print(f"  {'PASS' if ok else 'FAIL'}  {name:>14}: {value:.3f}  (>= {floor})")
        if not ok:
            failures.append(f"{name} {value:.3f} < {floor}")

    if failures:
        print("\nREGRESSION: " + "; ".join(failures))
        return 1
    print("\nAll retrieval-quality thresholds met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
