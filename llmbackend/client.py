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
          - Gemini: display_name (system instructions via config.extra["system"])
        """
        prompts = list(inputs)
        cfg = GenerationConfig.from_maybe_dict(config) if config is not None else None
        return self._provider.generate_batch(
            prompts, config=cfg, structured=schema, batch_options=batch_options
        )

    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get normalized status for a batch job.

        Returns a dict with:
          - state: normalized state ("pending", "in_progress", "completed", "failed", "cancelled")
          - provider: provider name
          - raw_state: original provider-specific state
          - batch_id: the batch identifier

        Raises RuntimeError if batch retrieval is not supported by the provider.
        """
        handle = self._provider._make_handle(batch_id)
        raw_state = self._provider._batch_status(handle)
        normalized = self._provider._normalize_batch_state(raw_state)
        return {
            "state": normalized,
            "provider": self.provider,
            "raw_state": raw_state,
            "batch_id": batch_id,
        }

    def fetch_batch_results(
        self,
        batch_id: str,
        schema: Optional[StructuredOutput] = None,
    ) -> List[Any]:
        """Fetch results from a completed batch job.

        Args:
            batch_id: The batch identifier returned from submit_batch()
            schema: Optional StructuredOutput to re-parse results (if not stored in handle)

        Returns:
            List of results in custom_id order (or submission order if no custom_ids)

        Raises:
            RuntimeError: If batch is not completed or retrieval not supported
            ValueError: If results cannot be parsed
        """
        # Reconstruct handle with schema metadata if provided
        meta = {"structured": schema} if schema else {}
        handle = self._provider._make_handle(batch_id, meta=meta)
        return self._provider._batch_results(handle)


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


def get_batch_status(
    batch_id: str,
    provider: str,
    model: str,
    **provider_options: Any,
) -> Dict[str, Any]:
    """Get batch status without recreating the original client.

    Example:
        status = llmbackend.get_batch_status("batch_abc123", "openai", "gpt-4o-mini")
        if status["state"] == "completed":
            results = llmbackend.fetch_batch_results("batch_abc123", "openai", "gpt-4o-mini")
    """
    c = client(provider=provider, model=model, **provider_options)
    return c.get_batch_status(batch_id)


def fetch_batch_results(
    batch_id: str,
    provider: str,
    model: str,
    schema: Optional[StructuredOutput] = None,
    **provider_options: Any,
) -> List[Any]:
    """Fetch batch results without recreating the original client.

    Example:
        results = llmbackend.fetch_batch_results(
            "batch_abc123",
            "openai",
            "gpt-4o-mini",
            schema=StructuredOutput(pydantic_model=MyModel),
        )
    """
    c = client(provider=provider, model=model, **provider_options)
    return c.fetch_batch_results(batch_id, schema=schema)
