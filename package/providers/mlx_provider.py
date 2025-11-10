from __future__ import annotations

from typing import Any, Optional

from .base import BaseProvider
from .util import require, map_config
from ..types import GenerationConfig, StructuredOutput


class MLXProvider(BaseProvider):
    """Local text generation via Apple's MLX (mlx-lm)."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        load = require("mlx_lm", "Install `mlx-lm` to use the 'mlx' provider.").load
        self._model, self._tokenizer = load(self.model, **self.options.get("load_kwargs", {}))

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        structured: Optional[StructuredOutput] = None,
    ) -> Any:
        if structured is not None:
            raise ValueError("Structured outputs are only supported for remote providers: openai, anthropic, gemini.")
        mlx_generate = require("mlx_lm", "Install `mlx-lm`.").generate
        cfg = config or GenerationConfig()
        kwargs = map_config(cfg, {
            "max_tokens": "max_tokens",
            "temperature": "temp",
            "top_p": "top_p",
        })

        text = mlx_generate(self._model, self._tokenizer, prompt, **kwargs)
        return text

    def generate_batch(self, prompts, config=None, structured=None, batch_options=None):  # type: ignore[override]
        raise ValueError(
            "Batch generation is only supported for remote providers: openai, anthropic, gemini."
        )
