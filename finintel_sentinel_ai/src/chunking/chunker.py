from __future__ import annotations

"""
Splits law sections into overlapping chunks, keeping each chunk's citation.

Most sections are short (under 1,000 characters) and stay whole. Long ones are
cut at sentence ends into ~1,000-character pieces that overlap by ~200
characters, so a rule that runs across a cut is still found in one piece.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

CHUNK_CHARS = 1000
OVERLAP_CHARS = 200

# Kept with every chunk, so an answer can always say where it came from.
METADATA_FIELDS = ["citation", "section", "title", "jurisdiction", "source", "source_url", "version_date"]


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)


def load_law_documents(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    docs = data["documents"]
    for doc in docs:
        missing = [f for f in ("doc_id", "citation", "text") if not doc.get(f)]
        if missing:
            raise ValueError(f"Law document {doc.get('doc_id', '?')} is missing {missing}")
    return docs


def split_text(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> List[str]:
    """Pieces of at most ~size characters, cut after a sentence end where possible."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= size:
        return [text]
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(". ", start + size // 2, end)   # last sentence end in the second half
            if cut != -1:
                end = cut + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def chunk_documents(docs: List[dict]) -> List[Chunk]:
    chunks = []
    for doc in docs:
        meta = {f: str(doc.get(f, "")) for f in METADATA_FIELDS}
        meta["doc_id"] = doc["doc_id"]
        pieces = split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunks.append(Chunk(
                chunk_id=f"{doc['doc_id']}#{i}",
                text=piece,
                metadata={**meta, "part": f"{i + 1} of {len(pieces)}"},
            ))
    return chunks
