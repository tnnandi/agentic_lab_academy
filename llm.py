"""Utilities for interacting with local or remote LLM backends."""

from __future__ import annotations

import asyncio
import functools
import os
from dataclasses import dataclass
from typing import Any, Tuple

import requests
from openai import OpenAI

try:
    from .config import LLM_CONFIG
    from .alcf_inference.inference_auth_token import get_access_token
except ImportError:
    from config import LLM_CONFIG
    from alcf_inference.inference_auth_token import get_access_token

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
    """Synchronously query the configured LLM backend."""

    global _total_tokens_used

    source = LLM_CONFIG.get("source", "ollama")
    model_name = model or LLM_CONFIG["default_model"]
    request_temperature = temperature if temperature is not None else LLM_CONFIG["temperature"].get(
        "research", 0.7
    )

    if source == "ollama":
        text, tokens = _query_ollama(prompt, model_name, request_temperature)
    elif source in {"alcf_sophia", "alcf_metis"}:
        text, tokens = _query_alcf(prompt, model_name, request_temperature, source)
    else:
        raise ValueError(f"Unsupported LLM source: {source}")

    _total_tokens_used += tokens
    _output_log.append(LLMResponse(prompt=prompt, response=text, model=model_name, tokens_used=tokens))
    return text


def _query_ollama(prompt: str, model_name: str, temperature: float) -> Tuple[str, int]:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False,
    }

    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data: dict[str, Any] = response.json()

    text = data.get("response", "").strip()
    tokens = int(data.get("eval_count", 0))

    return text, tokens


def _query_alcf(
    prompt: str,
    model_name: str,
    temperature: float,
    source: str,
) -> Tuple[str, int]:
    """Query the ALCF inference endpoint (Sophia or Metis)."""

    access_token = get_access_token()
    if source == "alcf_sophia":
        base_url = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
    elif source == "alcf_metis":
        base_url = "https://inference-api.alcf.anl.gov/resource_server/metis/api/v1"
    else:
        raise ValueError(f"Unsupported ALCF source: {source}")

    client = OpenAI(
        api_key=access_token,
        base_url=base_url,
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )

    text = response.choices[0].message.content.strip()
    usage = getattr(response, "usage", None)
    tokens = int(getattr(usage, "total_tokens", 0)) if usage else 0
    return text, tokens


async def query_llm_async(prompt: str, model: str | None = None, temperature: float | None = None) -> str:
    """Asynchronously query the LLM by delegating to a background thread."""

    loop = asyncio.get_running_loop()
    partial = functools.partial(query_llm, prompt, model=model, temperature=temperature)
    return await loop.run_in_executor(None, partial)


def get_total_tokens_used() -> int:
    return _total_tokens_used


def get_output_log() -> list[LLMResponse]:
    return list(_output_log)
