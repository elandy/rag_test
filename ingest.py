from services.vector_store import VectorStore
from data.docs import DOCUMENTS

vs = VectorStore()

for doc in DOCUMENTS:
    emb = vs.embed(doc["text"])
    vs.upsert_document(doc, emb)