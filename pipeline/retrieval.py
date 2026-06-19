import logging
import re

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "hows", "in", "is", "it", "of", "on", "or", "the", "to",
    "what", "where", "who",
}


class Retriever:
    def __init__(self, vector_store, min_distance: float = 0.6):
        """
        min_distance:
            cosine distance threshold (0 = perfect match, 1 = unrelated)
        """
        self.vector_store = vector_store
        self.min_distance = min_distance

    # -------------------------
    # TEXT PROCESSING
    # -------------------------

    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b[a-zA-Z][a-zA-Z']+\b", text.lower())

    def content_tokens(self, text: str) -> set[str]:
        return {t for t in self.tokenize(text) if t not in STOPWORDS}

    def lexical_overlap(self, query: str, text: str) -> float:
        q = self.content_tokens(query)
        if not q:
            return 0.0
        t = self.content_tokens(text)
        return len(q & t) / len(q)

    def extract_entities(self, text: str):
        return re.findall(r"\b[A-Z][a-zA-Z]+\b", text)

    def entity_overlap(self, query: str, text: str) -> float:
        q_entities = {e.lower() for e in self.extract_entities(query)}
        if not q_entities:
            return 0.0
        t = text.lower()
        return sum(1 for e in q_entities if e in t) / len(q_entities)

    # -------------------------
    # SCORE NORMALIZATION
    # -------------------------

    def to_similarity(self, distance: float) -> float:
        """
        Convert cosine distance → similarity
        so higher = better everywhere in pipeline
        """
        sim = 1.0 - distance
        return max(0.0, min(1.0, sim))

    def score_doc(self, query: str, doc: dict, distance: float) -> float:
        sim = self.to_similarity(distance)

        lexical = self.lexical_overlap(query, doc["text"])
        entity = self.entity_overlap(query, doc["text"])

        return sim + (0.25 * lexical) + (0.2 * entity)

    # -------------------------
    # SEARCH HELPERS
    # -------------------------

    def keyword_search(self, entity: str):
        results = []
        for doc in self.vector_store.get_all_documents():
            if entity.lower() in doc["text"].lower():
                results.append((doc, 0.0))  # treat as perfect match (distance=0)
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

    # -------------------------
    # RETRIEVAL CORE
    # -------------------------

    def retrieve_docs(self, query: str, k: int = 2):
        docs = self.vector_store.search(query, k)
        logger.debug("Retrieval query=%s docs=%s", query, self.summarize_docs(docs))
        return docs

    def retrieve_docs_multi_hop(self, query: str, k: int = 2):
        seen = set()

        def search_once(text: str, limit: int):
            key = (text.strip().lower(), limit)
            if key in seen:
                return []
            seen.add(key)
            return self.vector_store.search(text, k=limit)

        # 1. entity expansion
        query_entities = self.extract_entities(query)

        entity_docs = []
        for entity in query_entities[:3]:
            entity_docs.extend(self.keyword_search(entity))
            entity_docs.extend(search_once(entity, 2))

        # 2. semantic pass
        first_pass = search_once(query, 3)

        # 3. second hop
        doc_entities = set()
        for doc, _ in first_pass:
            doc_entities.update(self.extract_entities(doc["text"]))

        second_pass = []
        for entity in list(doc_entities)[:3]:
            second_pass.extend(search_once(entity, 1))

        # 4. merge
        all_docs = {}

        def add(docs, boost=0.0):
            for doc, distance in docs:
                score = self.score_doc(query, doc, distance + boost)
                doc_id = doc["id"]

                if doc_id not in all_docs or score > all_docs[doc_id][1]:
                    all_docs[doc_id] = (doc, score)

        add(entity_docs, 0.05)
        add(first_pass, 0.0)
        add(second_pass, 0.1)

        # 5. FINAL RANKING (HIGHER IS BETTER)
        ranked = sorted(all_docs.values(), key=lambda x: x[1], reverse=True)

        logger.debug("Ranked docs=%s", self.summarize_docs(ranked))
        print("SEARCH RESULTS:", ranked[:k])

        return ranked[:k]