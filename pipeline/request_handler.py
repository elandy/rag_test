import uuid

from flask import jsonify

from pipeline.generation import generate_answer
from pipeline.retrieval import Retriever
from pipeline.validation import parse_and_validate_request, ValidationError
from services.vector_store import VectorStore

vector_store = VectorStore()
retriever = Retriever(vector_store)

def handle_ask(request):
    request_id = str(uuid.uuid4())
    data = request.get_json()

    try:
        parsed = parse_and_validate_request(data)
    except ValidationError as e:
        return jsonify({"error": e.message}), e.status_code

    query = parsed["query"]
    k = parsed["k"]
    suspicious = parsed["suspicious"]

    # retrieval
    docs = retriever.retrieve_docs_multi_hop(query, k)

    if not docs:
        return jsonify({
            "question": query,
            "answer": "I don't know",
            "sources": [],
            "suspicious": suspicious,
            "request_id": request_id,
        })

    # generation
    answer = generate_answer(query, docs)

    return jsonify({
        "question": query,
        "answer": answer,
        "sources": [
            {
                "id": doc["id"],
                "source": doc["source"],
                "chunk_index": doc["chunk_index"],
                "text": doc["text"],
                "score": score,
            }
            for doc, score in docs
        ],
        "suspicious": suspicious,
        "request_id": request_id,
    })
