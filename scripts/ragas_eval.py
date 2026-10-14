"""Score the pipeline's answers with Ragas, using a local Ollama judge (D-21).

Why this is a separate script with its own environment
------------------------------------------------------
Ragas pulls a hosted-LLM stack (``langchain_openai`` / ``openai`` / ``langgraph``)
and pins langchain packages that are **incompatible with the service's validated
stack** — on this project's pins its current release does not even import
(``ModuleNotFoundError: langchain_community.chat_models.vertexai``), and forcing
it cascades into numpy/pyarrow conflicts. So Ragas never shares an interpreter
with the service. Instead the service env produces a JSONL seam::

    python -m eval --dump-ragas experiment/ragas_input.jsonl   # service env

and this script scores that file in a **dedicated Ragas env**::

    python -m venv .ragas && . .ragas/Scripts/activate      # Windows: .ragas\\Scripts\\activate
    pip install ragas langchain-ollama langchain-huggingface
    python scripts/ragas_eval.py experiment/ragas_input.jsonl

It keeps Ragas local-first (Ollama as judge, MiniLM embeddings — no hosted API,
no keys) and out of CI, where the deterministic retrieval gate (D-18) remains the
quality gate. Faithfulness / answer-relevancy / context-precision / context-recall
are LLM-judged and nondeterministic, which is why they are a local report, not a
build gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SETUP_HINT = (
    "Ragas is not importable in this environment. It conflicts with the service's\n"
    "pinned langchain stack, so run it in a dedicated venv (see this file's docstring\n"
    "and decision.md D-21):\n\n"
    "  python -m venv .ragas\n"
    "  .ragas\\Scripts\\activate            # macOS/Linux: source .ragas/bin/activate\n"
    "  pip install ragas langchain-ollama langchain-huggingface\n"
    "  python scripts/ragas_eval.py experiment/ragas_input.jsonl\n"
)


def _load(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Ragas scoring over the --dump-ragas JSONL seam.")
    ap.add_argument("input", nargs="?", default="experiment/ragas_input.jsonl",
                    help="JSONL produced by `python -m eval --dump-ragas`.")
    ap.add_argument("--model", default="llama3.2:3b", help="Ollama judge model.")
    ap.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL.")
    ap.add_argument("--embeddings", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Embedding model for answer-relevancy.")
    ap.add_argument("--out", default="experiment/ragas_results.csv", help="Where to write per-sample scores.")
    ap.add_argument("--limit", type=int, default=None, help="Score only the first N samples (judging is slow).")
    args = ap.parse_args()

    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy,
        )
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_ollama import ChatOllama
    except Exception as exc:  # noqa: BLE001 - any import failure means "wrong env"
        print(f"{_SETUP_HINT}\n(import error: {exc})", file=sys.stderr)
        return 2

    path = Path(args.input)
    if not path.exists():
        print(f"No such file: {path}\nRun `python -m eval --dump-ragas {path}` first "
              "(in the service env).", file=sys.stderr)
        return 2

    rows = _load(path, args.limit)
    if not rows:
        print(f"{path} is empty.", file=sys.stderr)
        return 2

    judge = LangchainLLMWrapper(ChatOllama(model=args.model, base_url=args.base_url, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=args.embeddings,
                              encode_kwargs={"normalize_embeddings": True})
    )
    metrics = [
        Faithfulness(llm=judge),
        ResponseRelevancy(llm=judge, embeddings=embeddings),
        LLMContextPrecisionWithReference(llm=judge),
        LLMContextRecall(llm=judge),
    ]

    print(f"Scoring {len(rows)} samples with Ragas (judge={args.model})…")
    dataset = EvaluationDataset.from_list(rows)
    result = evaluate(dataset=dataset, metrics=metrics, llm=judge, embeddings=embeddings)

    df = result.to_pandas()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")

    print("\n=== Ragas (local Ollama judge) ===")
    for col in ("faithfulness", "answer_relevancy", "llm_context_precision_with_reference",
                "context_recall"):
        if col in df.columns:
            print(f"  {col:>36}: {df[col].mean():.3f}")
    print(f"\nWrote per-sample scores to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
