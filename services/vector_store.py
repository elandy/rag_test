import hashlib
import json
import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
import requests

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "vector_index.json"
logger = logging.getLogger(__name__)


class IndexedChunk(TypedDict):
    id: str
    text: str
    source: str
    chunk_index: int
    embedding: list[float]


class VectorStore:
    def __init__(self, documents: list[dict]):
        self.index_path = INDEX_PATH
        self.documents = self._load_or_build_index(documents)
        self.query_embedding_cache: dict[str, np.ndarray] = {}
        self.search_cache: dict[tuple[str, int, float], list[tuple[IndexedChunk, float]]] = {}

    def _source_signature(self, documents: list[dict]) -> str:
        payload = json.dumps(documents, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_or_build_index(self, documents: list[dict]) -> list[IndexedChunk]:
        signature = self._source_signature(documents)

        if self.index_path.exists():
            with self.index_path.open(encoding="utf-8") as file:
                index_data = json.load(file)

            if index_data.get("signature") == signature:
                logger.info("Loaded vector index from %s", self.index_path)
                return index_data["documents"]

        indexed_documents = [
            {
                **document,
                "embedding": self.embed(document["text"]).tolist(),
            }
            for document in documents
        ]

        self.index_path.write_text(
            json.dumps(
                {
                    "signature": signature,
                    "documents": indexed_documents,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Built vector index at %s", self.index_path)
        return indexed_documents

    def embed(self, text: str) -> np.ndarray:
        response = requests.post(
            OLLAMA_EMBED_URL,
            json={
                "model": EMBED_MODEL,
                "prompt": text
            },
            timeout=10
        )

        if response.status_code != 200:
            raise Exception(f"Embedding error: {response.text}")

        data = response.json()
        return np.array(data["embedding"])

    def embed_query(self, text: str) -> np.ndarray:
        cache_key = text.strip().lower()
        cached = self.query_embedding_cache.get(cache_key)
        if cached is not None:
            return cached

        embedding = self.embed(text)
        self.query_embedding_cache[cache_key] = embedding
        return embedding

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def search(self, query: str, k: int = 2, threshold: float = 0.4):
        cache_key = (query.strip().lower(), k, threshold)
        cached = self.search_cache.get(cache_key)
        if cached is not None:
            return cached

        query_vec = self.embed_query(query)

        scores = []
        for i, document in enumerate(self.documents):
            doc_vec = np.array(document["embedding"])
            sim = self.cosine_similarity(query_vec, doc_vec)
            scores.append((i, sim))

        if not scores:
            return []

        scores.sort(key=lambda x: x[1], reverse=True)

        max_score = scores[0][1]

        # widen recall
        initial_k = max(k * 3, 5)
        top_candidates = scores[:initial_k]

        # filter
        filtered = [
            (i, score)
            for i, score in top_candidates
            if score >= max_score * 0.9 and score >= threshold
        ]

        # fallback
        if not filtered:
            filtered = [scores[0]]

        # final cap
        filtered = filtered[:k]

        results = [(self.documents[i], score) for i, score in filtered]
        self.search_cache[cache_key] = results
        return results
