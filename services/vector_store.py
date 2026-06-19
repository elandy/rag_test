import os
import logging
import numpy as np
import requests
import psycopg
from typing import TypedDict
from pgvector.psycopg import register_vector

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_EMBED_URL = f"{OLLAMA_HOST}/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

PG_DSN = os.getenv("PG_DSN")  # e.g. postgresql://user:pass@localhost:5432/rag

logger = logging.getLogger(__name__)


class IndexedChunk(TypedDict):
    id: str
    text: str
    source: str
    chunk_index: int


class VectorStore:
    def __init__(self):
        self.conn = psycopg.connect(PG_DSN)
        register_vector(self.conn)
        self.query_embedding_cache: dict[str, np.ndarray] = {}

    # -------------------------
    # EMBEDDINGS
    # -------------------------
    def embed(self, text: str) -> np.ndarray:
        response = requests.post(
            OLLAMA_EMBED_URL,
            json={
                "model": EMBED_MODEL,
                "prompt": text
            },
            timeout=30
        )

        if response.status_code != 200:
            raise RuntimeError(f"Embedding error: {response.text}")

        data = response.json()
        return np.array(data["embedding"], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        key = text.strip().lower()
        if key in self.query_embedding_cache:
            return self.query_embedding_cache[key]

        vec = self.embed(text)
        self.query_embedding_cache[key] = vec
        return vec

    # -------------------------
    # INSERT (optional but needed)
    # -------------------------
    def upsert_document(self, doc: IndexedChunk, embedding: np.ndarray):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (id, text, source, chunk_index, embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id)
                DO UPDATE SET
                    text = EXCLUDED.text,
                    source = EXCLUDED.source,
                    chunk_index = EXCLUDED.chunk_index,
                    embedding = EXCLUDED.embedding
                """,
                (
                    doc["id"],
                    doc["text"],
                    doc["source"],
                    doc["chunk_index"],
                    embedding
                )
            )
        self.conn.commit()

    # -------------------------
    # SEARCH (MAIN CHANGE)
    # -------------------------
    def search(self, query: str, k: int = 5):
        query_vec = self.embed_query(query)

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    text,
                    source,
                    chunk_index,
                    1 - (embedding <=> %s) AS similarity
                FROM documents
                ORDER BY embedding <#> %s
                LIMIT %s
                """,
                (query_vec, query_vec, k)
            )

            rows = cur.fetchall()

        results = [
            (
                {
                    "id": r[0],
                    "text": r[1],
                    "source": r[2],
                    "chunk_index": r[3],
                },
                r[4],
            )
            for r in rows
        ]

        return results

    def get_all_documents(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT id, text, source, chunk_index FROM documents")
            rows = cur.fetchall()

        return [
            {
                "id": r[0],
                "text": r[1],
                "source": r[2],
                "chunk_index": r[3],
            }
            for r in rows
        ]

    # -------------------------
    # OPTIONAL: CLOSE CONNECTION
    # -------------------------
    def close(self):
        self.conn.close()