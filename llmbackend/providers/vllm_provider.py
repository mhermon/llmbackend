from __future__ import annotations

from typing import Any, Iterable, List, Optional

from .base import BaseProvider, CompletedBatch
from .util import require, map_config
from ..types import GenerationConfig, StructuredOutput


class VLLMProvider(BaseProvider):
    """Local text generation via vLLM Python API."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        module = require("vllm", "Install `vllm` to use the 'vllm' provider.")
        self._SamplingParams = getattr(module, "SamplingParams")
        LLM = getattr(module, "LLM")
        self._llm = LLM(model=self.model, **self.options.get("llm_kwargs", {}))

    def _sampling_params(self, cfg: GenerationConfig) -> Any:
        params = map_config(
            cfg,
            {
                "temperature": "temperature",
                "top_p": "top_p",
                "max_tokens": "max_tokens",
                "seed": "seed",
            },
            include_extra=False,
        )
        extras = dict(getattr(cfg, "extra", {}) or {})
        params.update(extras)
        return self._SamplingParams(**params)

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        structured: Optional[StructuredOutput] = None,
    ) -> Any:
        if structured is not None:
            raise ValueError("Structured outputs are only supported for remote providers: openai, anthropic, gemini.")
        cfg = config or GenerationConfig()
        sp = self._sampling_params(cfg)
        outputs = self._llm.generate([prompt], sp)
        return self._extract_text(outputs, 0)

    def generate_batch(  # type: ignore[override]
        self,
        prompts: Iterable[str],
        config: Optional[GenerationConfig] = None,
        structured: Optional[StructuredOutput] = None,
        batch_options: Optional[dict[str, Any]] = None,
    ) -> CompletedBatch:
        if structured is not None:
            raise ValueError("Structured outputs are only supported for remote providers: openai, anthropic, gemini.")
        cfg = config or GenerationConfig()
        prompt_list = list(prompts)
        if not prompt_list:
            return CompletedBatch([])
        sp = self._sampling_params(cfg)
        outputs = self._llm.generate(prompt_list, sp)
        texts = [self._extract_text(outputs, i) for i in range(len(prompt_list))]
        return CompletedBatch(texts)

    def _extract_text(self, outputs: List[Any], idx: int) -> str:
        if not outputs or idx >= len(outputs):
            return ""
        choice_list = getattr(outputs[idx], "outputs", None)
        if not choice_list:
            return ""
        first = choice_list[0]
        return getattr(first, "text", "") or ""

