from pathlib import Path
from typing import TypedDict


DOCS_DIR = Path(__file__).with_name("docs")
CHUNK_SENTENCE_COUNT = 2
CHUNK_OVERLAP = 1


class DocumentChunk(TypedDict):
    id: str
    text: str
    source: str
    chunk_index: int


def _read_sentences(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _chunk_sentences(sentences: list[str], source: str) -> list[DocumentChunk]:
    if not sentences:
        return []

    step = max(1, CHUNK_SENTENCE_COUNT - CHUNK_OVERLAP)
    chunks: list[DocumentChunk] = []

    for chunk_index, start in enumerate(range(0, len(sentences), step)):
        window = sentences[start:start + CHUNK_SENTENCE_COUNT]
        if not window:
            continue

        chunks.append(
            {
                "id": f"{source}::chunk-{chunk_index}",
                "text": " ".join(window),
                "source": source,
                "chunk_index": chunk_index,
            }
        )

        if start + CHUNK_SENTENCE_COUNT >= len(sentences):
            break

    return chunks


def load_documents() -> list[DocumentChunk]:
    documents: list[DocumentChunk] = []

    for path in sorted(DOCS_DIR.glob("*.md")):
        documents.extend(_chunk_sentences(_read_sentences(path), path.name))

    return documents


DOCUMENTS = load_documents()
