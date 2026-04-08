import uuid

from flask import jsonify

from data.docs import DOCUMENTS
from pipeline.generation import generate_answer
from pipeline.retrieval import Retriever
from pipeline.validation import parse_and_validate_request, ValidationError
from services.vector_store import VectorStore

vector_store = VectorStore(DOCUMENTS)
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
    filtered_docs = retriever.filter_docs(docs)

    if not filtered_docs:
        return jsonify({
            "question": query,
            "answer": "I don't know",
            "sources": [],
            "suspicious": suspicious,
            "request_id": request_id,
        })

    # generation
    answer = generate_answer(query, filtered_docs)

    return jsonify({
        "question": query,
        "answer": answer,
        "sources": [{"text": d, "score": s} for d, s in filtered_docs],
        "suspicious": suspicious,
        "request_id": request_id,
    })