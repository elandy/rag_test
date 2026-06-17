import logging
import random
from functools import lru_cache
from time import sleep
from typing import Optional

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"
logger = logging.getLogger(__name__)

class RateLimitError(Exception):
    pass

MAX_RETRIES = 3

def generate_response(prompt: str) -> Optional[str]:
    backoff = 1

    for attempt in range(MAX_RETRIES):
        try:
            logger.debug("LLM generate attempt=%s", attempt + 1)
            return call_llm(prompt)

        except (RateLimitError, TimeoutError, ConnectionError, RuntimeError) as e:
            logger.warning("Retryable LLM error: %s. Retrying in %ss", e, backoff)
            sleep(backoff + random.uniform(0, 0.5))
            backoff *= 2

        except ValueError as e:
            logger.error("Non-retryable LLM error: %s", e)
            return None

    logger.error("LLM failed after retries")
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
