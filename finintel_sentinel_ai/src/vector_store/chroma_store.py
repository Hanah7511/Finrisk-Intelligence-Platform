from __future__ import annotations

"""
Stores law chunks and their vectors in ChromaDB, saved on disk, and searches them.
Vectors are made by our own Embedder and passed in, so Chroma never downloads
a model of its own and the index always matches the model in the metadata.
"""

from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings

from src.chunking.chunker import Chunk

COLLECTION = "aml_law"


class LawStore:
    def __init__(self, path: Path) -> None:
        # Telemetry off: nothing about this index is sent to Chroma.
        self._client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))

    def rebuild(self, chunks: List[Chunk], vectors: List[List[float]], model_name: str) -> int:
        """Replaces the whole collection, so a rebuild never leaves old chunks behind."""
        if COLLECTION in [c.name for c in self._client.list_collections()]:
            self._client.delete_collection(COLLECTION)
        col = self._client.create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine", "embedding_model": model_name}
        )
        col.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
            embeddings=vectors,
        )
        return col.count()

    def search(self, query_vector: List[float], k: int = 3, jurisdiction: Optional[str] = None) -> List[dict]:
        """Top k chunks by meaning; jurisdiction (e.g. "UAE") limits the search to that law."""
        col = self._client.get_collection(COLLECTION)
        where = {"jurisdiction": jurisdiction} if jurisdiction else None
        res = col.query(query_embeddings=[query_vector], n_results=k, where=where)
        return [
            {"text": doc, "metadata": meta, "similarity": round(1 - dist, 3)}
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def model_name(self) -> str:
        return self._client.get_collection(COLLECTION).metadata.get("embedding_model", "")
