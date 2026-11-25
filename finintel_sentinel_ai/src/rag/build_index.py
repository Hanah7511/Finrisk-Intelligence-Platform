from __future__ import annotations

"""
Builds the law index once: law texts -> chunks -> vectors -> ChromaDB.
Run from finintel_sentinel_ai:  python -m src.rag.build_index

Two law files:
- us_aml_regulations.json: US federal law, public domain, committed to the repo.
- aml_regulations_local.json: UAE Federal Decree-Law No. 10 of 2025 and FATF
  Recommendations. Used only if present on this computer; never committed
  (FATF text is copyrighted, the UAE translation's terms are not yet confirmed).
The index itself (vector_db/) holds the text too, so it is never committed either.
"""

import logging
from collections import Counter
from pathlib import Path

from src.chunking.chunker import chunk_documents, load_law_documents
from src.embeddings.embedder import Embedder
from src.vector_store.chroma_store import LawStore

ROOT = Path(__file__).resolve().parents[2]                    # finintel_sentinel_ai/
KNOWLEDGE_BASE = ROOT.parent / "data" / "knowledge_base"
PUBLIC_LAW_FILE = KNOWLEDGE_BASE / "us_aml_regulations.json"
LOCAL_LAW_FILE = KNOWLEDGE_BASE / "aml_regulations_local.json"
INDEX_DIR = ROOT / "vector_db"

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    docs = load_law_documents(PUBLIC_LAW_FILE)
    if LOCAL_LAW_FILE.exists():
        docs += load_law_documents(LOCAL_LAW_FILE)
    else:
        logger.info("No local law file (%s): indexing US law only.", LOCAL_LAW_FILE.name)
    chunks = chunk_documents(docs)
    logger.info("Loaded %d law sections -> %d chunks", len(docs), len(chunks))

    embedder = Embedder()
    vectors = embedder.embed([c.text for c in chunks])
    count = LawStore(INDEX_DIR).rebuild(chunks, vectors, embedder.model_name)

    by_place = Counter(d.get("jurisdiction", "?") for d in docs)
    print("\n--- Law Index Report -------------------------")
    print(f"Law sections        : {len(docs)}  {dict(by_place)}")
    print(f"Chunks stored       : {count}")
    print(f"Embedding model     : {embedder.model_name} ({len(vectors[0])} numbers per chunk)")
    print(f"Saved to            : {INDEX_DIR}")
    print("------------------------------------------------\n")


if __name__ == "__main__":
    main()
