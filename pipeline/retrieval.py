import logging
import re


logger = logging.getLogger(__name__)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "hows",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "where",
    "who",
}


class Retriever:
    def __init__(self, vector_store, min_score: float = 0.3):
        self.vector_store = vector_store
        self.min_score = min_score

    def filter_docs(self, query: str, docs):
        if not docs:
            return []

        top_score = docs[0][1]
        query_has_entities = bool(self.extract_entities(query))
        filtered = []

        for doc, score in docs:
            lexical_score = self.lexical_overlap(query, doc["text"])
            entity_score = self.entity_overlap(query, doc["text"])

            if score < self.min_score:
                continue

            if query_has_entities and entity_score == 0 and score < top_score * 0.9:
                continue

            if score < top_score * 0.75 and lexical_score < 0.2 and entity_score == 0:
                continue

            filtered.append((doc, score))

        return filtered

    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b[a-zA-Z][a-zA-Z']+\b", text.lower())

    def content_tokens(self, text: str) -> set[str]:
        return {token for token in self.tokenize(text) if token not in STOPWORDS}

    def lexical_overlap(self, query: str, text: str) -> float:
        query_tokens = self.content_tokens(query)
        if not query_tokens:
            return 0.0

        text_tokens = self.content_tokens(text)
        overlap = query_tokens & text_tokens
        return len(overlap) / len(query_tokens)

    def extract_entities(self, text: str):
        # naive: capitalized words
        return re.findall(r"\b[A-Z][a-zA-Z]+\b", text)

    def entity_overlap(self, query: str, text: str) -> float:
        query_entities = {entity.lower() for entity in self.extract_entities(query)}
        if not query_entities:
            return 0.0

        text_lower = text.lower()
        matches = sum(1 for entity in query_entities if entity in text_lower)
        return matches / len(query_entities)

    def score_doc(self, query: str, doc: dict, base_score: float) -> float:
        lexical_score = self.lexical_overlap(query, doc["text"])
        entity_score = self.entity_overlap(query, doc["text"])
        return base_score + (0.25 * lexical_score) + (0.2 * entity_score)

    def keyword_search(self, entity: str):
        results = []
        for doc in self.vector_store.documents:
            if entity.lower() in doc["text"].lower():
                results.append((doc, 1.0))  # high confidence
        return results

    def summarize_docs(self, docs):
        return [
            {
                "id": doc["id"],
                "source": doc["source"],
                "score": round(float(score), 4),
            }
            for doc, score in docs
        ]

    # ---- Retrieval ----
    def retrieve_docs(self, query: str, k: int = 2):
        docs = self.vector_store.search(query, k)
        logger.debug("Retrieval query=%s docs=%s", query, self.summarize_docs(docs))
        return docs

    def retrieve_docs_multi_hop(self, query: str, k: int = 2):
        seen_searches = set()

        def search_once(text: str, limit: int):
            cache_key = (text.strip().lower(), limit)
            if cache_key in seen_searches:
                return []
            seen_searches.add(cache_key)
            return self.vector_store.search(text, k=limit)

        # ---- 1. ENTITY-FIRST RETRIEVAL ----
        query_entities = self.extract_entities(query)
        logger.debug("Multi-hop query=%s query_entities=%s", query, query_entities)
        entity_docs = []
        for entity in query_entities[:3]:  # limit expansion
            # keyword first (exact match)
            entity_docs.extend(self.keyword_search(entity))

            # semantic fallback
            entity_docs.extend(search_once(entity, limit=2))
        logger.debug("Entity-first docs=%s", self.summarize_docs(entity_docs))

        # ---- 2. FIRST PASS (SEMANTIC) ----
        first_pass = search_once(query, limit=3)
        logger.debug("First-pass docs=%s", self.summarize_docs(first_pass))

        # ---- 3. MULTI-HOP EXPANSION ----
        doc_entities = set()
        for doc, _ in first_pass:
            doc_entities.update(self.extract_entities(doc["text"]))

        second_pass = []
        for entity in list(doc_entities)[:3]:  # limit again
            second_pass.extend(search_once(entity, limit=1))
        logger.debug("Second-pass docs=%s", self.summarize_docs(second_pass))

        # ---- 4. MERGE + DEDUPE ----
        all_docs = {}

        def add_docs(docs, boost=0.0):
            for doc, score in docs:
                score = self.score_doc(query, doc, score + boost)
                doc_id = doc["id"]
                existing = all_docs.get(doc_id)
                if existing is None or score > existing[1]:
                    all_docs[doc_id] = (doc, score)

        add_docs(entity_docs, boost=0.05)  # entity signal
        add_docs(first_pass, boost=0.0)  # base
        add_docs(second_pass, boost=0.1)  # multi-hop signal

        # ---- 5. FINAL RANKING ----
        ranked = sorted(all_docs.values(), key=lambda x: x[1], reverse=True)
        logger.debug("Ranked docs=%s", self.summarize_docs(ranked))

        return ranked[:k]
