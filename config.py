"""Configuration for the Academy-powered Agentic Lab."""

MAX_ROUNDS = 2

LLM_CONFIG = {
    "default_model": "gpt-oss:20b",
    # "default_model": "llama3.1:8b",
    "temperature": {
        "research": 0.3,
        "coding": 0.2,
        "critic": 0.4,
        "execution": 0.1,
        "review": 0.1,
    },
}

__all__ = ["MAX_ROUNDS", "LLM_CONFIG"]
