from __future__ import annotations

"""
Asks the law index a question and prints the closest law chunks with citations.
Run from finintel_sentinel_ai:
  python -m src.rag.query "When must a bank file a suspicious activity report?"
  python -m src.rag.query --in UAE "When must suspicious transactions be reported?"
--in limits the search to one jurisdiction: US, UAE or International (FATF).
"""

import argparse
import sys
from typing import List, Optional

from src.embeddings.embedder import Embedder
from src.rag.build_index import INDEX_DIR
from src.vector_store.chroma_store import LawStore

PREVIEW_CHARS = 300


def retrieve(question: str, k: int = 3, jurisdiction: Optional[str] = None) -> List[dict]:
    store = LawStore(INDEX_DIR)
    embedder = Embedder()
    if store.model_name() != embedder.model_name:
        raise RuntimeError("The index was built with another model: run python -m src.rag.build_index")
    return store.search(embedder.embed([question])[0], k, jurisdiction)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # law text has dashes and quotes Windows consoles may not print
    parser = argparse.ArgumentParser(description="Search the AML law index.")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--in", dest="jurisdiction", choices=["US", "UAE", "International"])
    args = parser.parse_args()
    question = " ".join(args.question) or "When must a bank file a suspicious activity report?"

    where = f" (only {args.jurisdiction})" if args.jurisdiction else ""
    print(f"\nQuestion: {question}{where}\n")
    for rank, hit in enumerate(retrieve(question, jurisdiction=args.jurisdiction), start=1):
        m = hit["metadata"]
        text = hit["text"]
        preview = text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + " ..."
        print(f"{rank}. [{m['jurisdiction']}] {m['citation']}  (similarity {hit['similarity']}, part {m['part']})")
        print(f"   {m['title']}")
        print(f"   {preview}")
        print(f"   source: {m['source_url']}\n")


if __name__ == "__main__":
    main()
