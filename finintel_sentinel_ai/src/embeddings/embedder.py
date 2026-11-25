from __future__ import annotations

"""
Turns text into meaning vectors with a small local model (no API key, no cost).
all-MiniLM-L6-v2: 384 numbers per text. Vectors are normalised, so cosine
similarity compares meaning, not text length.
"""

import logging
import os
from typing import List

# Quiet model loading: no progress bars or load reports, no per-file HTTP log lines.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.getLogger("httpx").setLevel(logging.WARNING)

from sentence_transformers import SentenceTransformer  # noqa: E402

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
