import random
from functools import lru_cache
from time import sleep
from typing import Optional

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"

class RateLimitError(Exception):
    pass

MAX_RETRIES = 3

def generate_response(prompt: str) -> Optional[str]:
    backoff = 1

    for attempt in range(MAX_RETRIES):
        try:
            print(f"[LLM] Attempt {attempt + 1}")
            return call_llm(prompt)

        except (RateLimitError, TimeoutError, ConnectionError, RuntimeError) as e:
            print(f"[LLM] Retryable error: {e}. Retrying in {backoff}s...")
            sleep(backoff + random.uniform(0, 0.5))
            backoff *= 2

        except ValueError as e:
            print(f"[LLM] Non-retryable error: {e}")
            return None

    print("[LLM] Failed after retries")
    return None

@lru_cache(maxsize=100)
def cached_generate(prompt: str):
    return generate_response(prompt)


def call_llm(prompt: str, timeout: int = 100) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=timeout
        )

    except requests.exceptions.Timeout:
        raise TimeoutError("LLM request timed out")

    except requests.exceptions.ConnectionError:
        raise ConnectionError("LLM connection failed")

    if response.status_code == 429:
        raise RateLimitError("Rate limit exceeded")

    if 500 <= response.status_code < 600:
        raise RuntimeError(f"Server error: {response.text}")

    if response.status_code != 200:
        # non-retryable
        raise ValueError(f"Bad request: {response.text}")

    data = response.json()
    return data.get("response", "")
