import re



class Retriever:
    def __init__(self, vector_store, min_score: float = 0.3):
        self.vector_store = vector_store
        self.min_score = min_score

    def filter_docs(self, docs):
        return [d for d in docs if d[1] >= self.min_score]

    def extract_entities(self, text: str):
        # naive: capitalized words
        return re.findall(r"\b[A-Z][a-zA-Z]+\b", text)

    def keyword_search(self, entity: str):
        results = []
        for doc in self.vector_store.documents:
            if entity.lower() in doc.lower():
                results.append((doc, 1.0))  # high confidence
        return results

    # ---- Retrieval ----
    def retrieve_docs(self, query: str, k: int = 2):
        docs = self.vector_store.search(query, k)
        print(f"[RETRIEVAL] Query: {query}")
        print(f"[RETRIEVAL] Docs: {docs}")
        return docs


    def retrieve_docs_multi_hop(self, query: str, k: int = 2):
        # ---- 1. ENTITY-FIRST RETRIEVAL ----
        query_entities = self.extract_entities(query)
        print(f"{query_entities=}")
        entity_docs = []
        for entity in query_entities[:3]:  # limit expansion
            # keyword first (exact match)
            entity_docs.extend(self.keyword_search(entity))

            # semantic fallback
            entity_docs.extend(self.vector_store.search(entity, k=2))
        print(f"{entity_docs=}")

        # ---- 2. FIRST PASS (SEMANTIC) ----
        first_pass = self.vector_store.search(query, k=3)
        print(f"{first_pass=}")

        # ---- 3. MULTI-HOP EXPANSION ----
        doc_entities = set()
        for text, _ in first_pass:
            doc_entities.update(self.extract_entities(text))

        second_pass = []
        for entity in list(doc_entities)[:3]:  # limit again
            second_pass.extend(self.vector_store.search(entity, k=1))
        print(f"{second_pass=}")

        # ---- 4. MERGE + DEDUPE ----
        all_docs = {}

        def add_docs(docs, boost=0.0):
            for doc, score in docs:
                score += boost
                all_docs[doc] = max(all_docs.get(doc, 0), score)

        add_docs(entity_docs, boost=0.05)  # entity signal
        add_docs(first_pass, boost=0.0)  # base
        add_docs(second_pass, boost=0.1)  # multi-hop signal

        # ---- 5. FINAL RANKING ----
        ranked = sorted(all_docs.items(), key=lambda x: x[1], reverse=True)

        return ranked[:k]