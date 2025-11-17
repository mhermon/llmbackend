from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .types import GenerationConfig, StructuredOutput
from .providers.base import BaseProvider, Batch


def _provider_registry():
    # Lazy import to keep import-time light when SDKs are missing
    return {
        "openai": _load("llmbackend.providers.openai_provider", "OpenAIProvider"),
        "anthropic": _load("llmbackend.providers.anthropic_provider", "AnthropicProvider"),
        "gemini": _load("llmbackend.providers.gemini_provider", "GeminiProvider"),
        "transformers": _load("llmbackend.providers.transformers_provider", "TransformersProvider"),
        "vllm": _load("llmbackend.providers.vllm_provider", "VLLMProvider"),
        "mlx": _load("llmbackend.providers.mlx_provider", "MLXProvider"),
    }


def _load(module: str, name: str):
    import importlib

    def factory():
        mod = importlib.import_module(module)
        return getattr(mod, name)

    return factory


@dataclass
class Client:
    """Provider-agnostic LLM client for text-only generation.

    Parameters
    ----------
    provider: str
        One of: "openai", "anthropic", "gemini", "transformers", "vllm", "mlx".
    model: str
        Model identifier (e.g., "gpt-4o-mini", HF repo id, local path).
    provider_options: dict
        Credentials or provider-specific options (e.g., api_key, base_url, model_kwargs).
    """

    provider: str
    model: str
    provider_options: Dict[str, Any] | None = None

    def __post_init__(self):
        reg = _provider_registry()
        if self.provider not in reg:
            raise ValueError(
                f"Unknown provider '{self.provider}'. Valid: {', '.join(sorted(reg.keys()))}"
            )
        ProviderCls = reg[self.provider]()
        self._provider: BaseProvider = ProviderCls(self.model, **(self.provider_options or {}))

    def get_response(
        self,
        input: str,
        config: Optional[GenerationConfig | Dict[str, Any]] = None,
        schema: Optional[StructuredOutput] = None,
    ) -> Any:
        """Generate a single response for the given `input`.

        - Returns a string by default.
        - If `schema` is provided, returns parsed structured output.
        """
        cfg = GenerationConfig.from_maybe_dict(config) if config is not None else None
        return self._provider.generate(input, config=cfg, structured=schema)

    def submit_batch(
        self,
        inputs: Iterable[str],
        config: Optional[GenerationConfig | Dict[str, Any]] = None,
        schema: Optional[StructuredOutput] = None,
        batch_options: Optional[Dict[str, Any]] = None,
    ) -> Batch:
        """Generate responses for multiple prompts using batch strategy.

        - Remote providers:
          - OpenAI, Gemini: return a `Batch` handle with `.id`, `.status()`, `.results()`.
          - Anthropic: returns a completed `Batch` (immediate results).
        - Local providers:
          - transformers, vllm, mlx: return a completed `Batch` with results immediately.

        `batch_options` may include provider-specific fields, e.g.:
          - OpenAI: display_name, completion_window, custom_ids, system
          - Gemini: display_name (system_instruction lives in config.extra)
        """
        prompts = list(inputs)
        cfg = GenerationConfig.from_maybe_dict(config) if config is not None else None
        return self._provider.generate_batch(
            prompts, config=cfg, structured=schema, batch_options=batch_options
        )


# Convenience top-level functions
def client(provider: str, model: str, **provider_options: Any) -> Client:
    """Create a Client with the given provider and model.

    Example
    -------
    client = llmbackend.client(provider="openai", model="gpt-4o-mini", api_key="...")
    """
    return Client(provider=provider, model=model, provider_options=provider_options or None)


def get_response(
    provider: str,
    model: str,
    input: str,
    config: Optional[GenerationConfig | Dict[str, Any]] = None,
    schema: Optional[StructuredOutput] = None,
    **provider_options: Any,
) -> Any:
    """One-off generate without creating a Client explicitly."""
    c = client(provider=provider, model=model, **provider_options)
    return c.get_response(input=input, config=config, schema=schema)


def create_batch(
    inputs: Iterable[str],
    provider: str,
    model: str,
    config: Optional[GenerationConfig | Dict[str, Any]] = None,
    schema: Optional[StructuredOutput] = None,
    batch_options: Optional[Dict[str, Any]] = None,
    **provider_options: Any,
) -> Batch:
    """One-off batch generation function.

    Returns a provider-specific Batch handle. For remote providers (OpenAI,
    Gemini), this wraps a remote job. For Anthropic, it's already completed.
    """
    c = client(provider=provider, model=model, **provider_options)
    return c.submit_batch(inputs=inputs, config=config, schema=schema, batch_options=batch_options)
