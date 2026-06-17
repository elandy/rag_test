# ---- LLM (mock) ----
from typing import Tuple

from services.llm_client import cached_generate


def generate_answer(query: str, docs: list[Tuple[dict, float]]) -> str:
    context = "\n\n".join(
        (
            f"[Doc {i + 1}] "
            f"(source={doc['source']}, chunk={doc['chunk_index']})\n"
            f"{doc['text']}"
        )
        for i, (doc, _) in enumerate(docs)
    )

    prompt = f"""
    You are a question-answering system.
    Instructions:
        - Use ALL relevant context to answer the question
        - Combine information across documents if needed
        - If the answer requires multiple steps, reason through them
        - If the answer is not in the context, say "I don't know".
        - Do not make up information.

    Context:
    <<<
    {context}
    >>>

    Question:
    {query}

    Answer:
    """.strip()

    response = cached_generate(prompt)

    if response is None:
        return "Error: LLM service unavailable"

    return response
