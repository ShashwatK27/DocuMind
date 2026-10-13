"""Download the evaluation corpus (public arXiv papers) into data/documents/.

Run from the repo root:  python scripts/fetch_corpus.py

These third-party PDFs are not committed to the repo; this script fetches them
so the multi-document evaluation (`python -m eval`) is reproducible. The golden
set in eval/golden_set.py is grounded against these exact files.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

# filename in data/documents/  ->  source URL
CORPUS = {
    "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762",
    "bert_devlin_2019.pdf": "https://arxiv.org/pdf/1810.04805",
    "retrieval_augmented_generation_lewis_2020.pdf": "https://arxiv.org/pdf/2005.11401",
}

DEST = Path(__file__).resolve().parent.parent / "data" / "documents"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, url in CORPUS.items():
        target = DEST / name
        if target.exists():
            print(f"skip   {name} (already present)")
            continue
        print(f"fetch  {name} <- {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        if not data.startswith(b"%PDF"):
            raise RuntimeError(f"{url} did not return a PDF")
        target.write_bytes(data)
        print(f"       wrote {len(data):,} bytes")
    print("done.")


if __name__ == "__main__":
    main()
