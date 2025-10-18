"""Utilities for interacting with the local Ollama endpoint."""

from __future__ import annotations

import asyncio
import functools
import os
from dataclasses import dataclass
from typing import Any

import requests

try:
    from .config import LLM_CONFIG
except ImportError:
    from config import LLM_CONFIG

__all__ = ["query_llm", "query_llm_async", "LLMResponse"]

os.environ.setdefault("NO_PROXY", "localhost")


@dataclass
class LLMResponse:
    prompt: str
    response: str
    model: str
    tokens_used: int


_total_tokens_used: int = 0
_output_log: list[LLMResponse] = []


def query_llm(prompt: str, model: str | None = None, temperature: float | None = None) -> str:
    """Synchronously query the locally hosted LLM via the Ollama REST API."""

    global _total_tokens_used

    model_name = model or LLM_CONFIG["default_model"]
    request_temperature = temperature if temperature is not None else LLM_CONFIG["temperature"].get(
        "research", 0.7
    )

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": request_temperature,
        "stream": False,
    }

    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data: dict[str, Any] = response.json()

    text = data.get("response", "").strip()
    tokens = int(data.get("eval_count", 0))

    _total_tokens_used += tokens
    _output_log.append(LLMResponse(prompt=prompt, response=text, model=model_name, tokens_used=tokens))
    return text


async def query_llm_async(prompt: str, model: str | None = None, temperature: float | None = None) -> str:
    """Asynchronously query the LLM by delegating to a background thread."""

    loop = asyncio.get_running_loop()
    partial = functools.partial(query_llm, prompt, model=model, temperature=temperature)
    return await loop.run_in_executor(None, partial)


def get_total_tokens_used() -> int:
    return _total_tokens_used


def get_output_log() -> list[LLMResponse]:
    return list(_output_log)
