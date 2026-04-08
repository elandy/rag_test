from pipeline.security import is_suspicious

class ValidationError(Exception):
    def __init__(self, message, status_code=400):
        self.message = message
        self.status_code = status_code

def parse_and_validate_request(data):
    if not data or "question" not in data:
        raise ValidationError("Missing 'question'")

    query = data["question"]
    if not isinstance(query, str):
        raise ValidationError("Question must be a string")

    query = " ".join(query.strip().split())

    if len(query) < 5:
        raise ValidationError("Query too short")

    if len(query) > 1000:
        raise ValidationError("Query too long")

    k = data.get("k", 2)
    if not isinstance(k, int):
        raise ValidationError("k must be an integer")

    k = max(1, min(k, 5))
    suspicious = is_suspicious(query)
    return {
        "query": query,
        "k": k,
        "suspicious": suspicious,
    }