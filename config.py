"""Configuration for the Academy-powered Agentic Lab with flexible LLM backends."""

MAX_ROUNDS = 3
MAX_EXECUTION_ATTEMPTS = 5 # Number of loops between the code executor and the code writer agent

# Supported sources:
#   - "ollama": local Ollama REST API (default)
#   - "alcf_sophia": ALCF inference endpoint (Sophia/vLLM)
#   - "alcf_metis": ALCF inference endpoint (Metis/OpenAI-compatible hosted on Sambanova machines)
LLM_CONFIG = {
    # "source": "ollama",
    # "source": "alcf_sophia",
    # "source": "alcf_metis",
    "source": "openrouter",

    # "default_model": "openai/gpt-oss-120b", #"gpt-oss:20b",
    "default_model": "openrouter/polaris-alpha",

    # "max_prompt_chars": 20000,
    # "max_response_tokens": 1024,
    "temperature": {
        "research": 0.3,
        "coding": 0.2,
        "critic": 0.4,
        "execution": 0.1,
        "review": 0.1,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
    }
}

__all__ = ["MAX_ROUNDS", "LLM_CONFIG"]
