from __future__ import annotations

from typing import Any, Optional

from .base import BaseProvider
from .util import require, map_config, derive_do_sample
from ..types import GenerationConfig, StructuredOutput


class TransformersProvider(BaseProvider):
    """Local text generation via Hugging Face Transformers pipeline."""

    def _get_pipeline(self):
        pipeline = require("transformers", "Install `transformers` to use the 'transformers' provider.").pipeline
        model_kwargs = self.options.get("model_kwargs", {})
        return pipeline("text-generation", model=self.model, **model_kwargs)

    def _kwargs(self, cfg: GenerationConfig) -> dict[str, Any]:
        d: dict[str, Any] = {"return_full_text": False, "do_sample": derive_do_sample(cfg)}
        d.update(map_config(cfg, {
            "max_tokens": "max_new_tokens",
            "temperature": "temperature",
            "top_p": "top_p",
        }))
        return d

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        structured: Optional[StructuredOutput] = None,
    ) -> Any:
        if structured is not None:
            raise ValueError("Structured outputs are only supported for remote providers: openai, anthropic, gemini.")
        pipe = self._get_pipeline()
        cfg = config or GenerationConfig()
        out = pipe(prompt, **self._kwargs(cfg))
        text: str = out[0]["generated_text"] if out else ""
        return text

    def generate_batch(self, prompts, config=None, structured=None, batch_options=None):  # type: ignore[override]
        raise ValueError(
            "Batch generation is only supported for remote providers: openai, anthropic, gemini."
        )
