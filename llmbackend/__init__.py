"""Lightweight, provider-agnostic text LLM client.

This package exposes a tiny, consistent API for generating text from a
variety of LLM providers (local and remote), with both synchronous and
batched interfaces.

Quickstart
----------

    import llmbackend

    # One-off sync call
    resp = llmbackend.get_response(
        provider="openai",
        model="gpt-4o-mini",
        input="Write a haiku about oceans.",
        config={"temperature": 0.7, "max_tokens": 100},
    )

    # Client + batch (remote providers return a handle)
    client = llmbackend.client(provider="openai", model="gpt-4o-mini")
    batch = client.submit_batch([
        "One fish.",
        "Two fish.",
    ], batch_options={"display_name": "demo"})
    print(batch.id)
    # later: print(batch.status()); results = batch.results()

Environment variables for remote providers:
    - OpenAI:      OPENAI_API_KEY, OPENAI_BASE_URL (optional)
    - Anthropic:   ANTHROPIC_API_KEY
    - Google:      GOOGLE_API_KEY

Providers:
    - openai, anthropic, gemini (remote)
    - transformers (Hugging Face local), vllm, mlx (local)

Notes:
    - Structured outputs support JSON Schema or Pydantic models on remote
      providers (OpenAI, Anthropic, Gemini) only. Local providers return text
      and do not support structured outputs.
    - Batch generation: OpenAI and Gemini return a handle with status/results; Anthropic returns a completed handle; Local providers (transformers, vllm, mlx) return a completed handle immediately.
"""

from .client import Client, client, get_response, create_batch, get_batch_status, fetch_batch_results
from .providers.base import Batch
from .types import GenerationConfig, StructuredOutput

__all__ = [
    "Client",
    "client",
    "get_response",
    "create_batch",
    "get_batch_status",
    "fetch_batch_results",
    "Batch",
    "GenerationConfig",
    "StructuredOutput",
]
