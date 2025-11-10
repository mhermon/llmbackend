from __future__ import annotations

from typing import Any, Optional

from .base import BaseProvider
from .util import require, map_config
from ..types import GenerationConfig, StructuredOutput


class VLLMProvider(BaseProvider):
    """Local text generation via vLLM Python API."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        LLM = require("vllm", "Install `vllm` to use the 'vllm' provider.").LLM
        self._llm = LLM(model=self.model, **self.options.get("llm_kwargs", {}))

    def _sampling(self, cfg: GenerationConfig) -> dict[str, Any]:
        return map_config(cfg, {
            "temperature": "temperature",
            "top_p": "top_p",
            "max_tokens": "max_tokens",
            "seed": "seed",
        })

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        structured: Optional[StructuredOutput] = None,
    ) -> Any:
        if structured is not None:
            raise ValueError("Structured outputs are only supported for remote providers: openai, anthropic, gemini.")
        SamplingParams = require("vllm", "Install `vllm`.").SamplingParams
        cfg = config or GenerationConfig()
        sp = SamplingParams(**self._sampling(cfg))
        outputs = self._llm.generate([prompt], sp)
        text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        return text

    def generate_batch(self, prompts, config=None, structured=None, batch_options=None):  # type: ignore[override]
        raise ValueError(
            "Batch generation is only supported for remote providers: openai, anthropic, gemini."
        )
