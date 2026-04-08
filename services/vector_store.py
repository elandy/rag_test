import requests
import numpy as np

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"


class VectorStore:
    def __init__(self, documents: list[str]):
        self.documents = documents
        self.embeddings = [self.embed(doc) for doc in documents]

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

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def search(self, query: str, k: int = 2, threshold: float = 0.4):
        query_vec = self.embed(query)

        scores = []
        for i, doc_vec in enumerate(self.embeddings):
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

        return [(self.documents[i], score) for i, score in filtered]