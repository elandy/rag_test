SUSPICIOUS_PATTERNS = [
    "ignore previous instructions",
    "system prompt",
    "reveal",
    "confidential",
]

def is_suspicious(query: str) -> bool:
    q = query.lower()
    return any(p in q for p in SUSPICIOUS_PATTERNS)