from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Dict, Union

from .base import BaseProvider, CompletedBatch
from .util import require, map_config
from ..types import GenerationConfig, StructuredOutput

Conversation = Sequence[Dict[str, Any]]


class VLLMProvider(BaseProvider):
    """Local text generation via vLLM Python API.
    
    Supports both standard prompt generation and chat-based generation with LoRA adapters.
    
    Options:
        - llm_kwargs: dict with LLM initialization parameters (tokenizer, enable_lora, max_lora_rank, trust_remote_code, etc.)
        - lora_path: Path to LoRA adapter weights (if using LoRA)
        - lora_name: Name for the LoRA adapter (default: extracted from lora_path)
        - lora_rank: Rank of the LoRA adapter (default: 1)
    """

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        module = require("vllm", "Install `vllm` to use the 'vllm' provider.")
        self._SamplingParams = getattr(module, "SamplingParams")
        
        # Import LoRARequest - handle different vLLM versions
        try:
            from vllm.lora.request import LoRARequest
            self._LoRARequest = LoRARequest
        except (ImportError, AttributeError):
            # LoRA may not be available in older vLLM versions
            self._LoRARequest = None
        
        LLM = getattr(module, "LLM")
        
        # Build LLM kwargs with defaults
        llm_kwargs = dict(self.options.get("llm_kwargs", {}))
        llm_kwargs.setdefault("model", self.model)
        
        self._llm = LLM(**llm_kwargs)
        
        # Store LoRA configuration
        self._lora_path = self.options.get("lora_path")
        self._lora_name = self.options.get("lora_name")
        self._lora_rank = self.options.get("lora_rank", 1)

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
        # Extract vllm-specific extras (exclude conversation and lora_request)
        extras = dict(getattr(cfg, "extra", {}) or {})
        for key in ("conversation", "lora_path", "lora_name", "lora_rank"):
            extras.pop(key, None)
        params.update(extras)
        return self._SamplingParams(**params)

    def _make_lora_request(self, cfg: GenerationConfig) -> Any:
        """Create a LoRARequest from config or instance defaults."""
        if self._LoRARequest is None:
            return None
            
        extras = getattr(cfg, "extra", {}) or {}
        lora_path = extras.get("lora_path") or self._lora_path
        if not lora_path:
            return None
        
        lora_name = extras.get("lora_name") or self._lora_name
        if not lora_name:
            # Derive name from path
            from pathlib import Path
            lora_name = Path(lora_path).name
        
        lora_rank = extras.get("lora_rank", self._lora_rank)
        return self._LoRARequest(lora_name, lora_rank, lora_path)

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
        
        # Check if conversation is provided in extras
        extras = getattr(cfg, "extra", {}) or {}
        conversation = extras.get("conversation")
        lora_request = self._make_lora_request(cfg)
        
        # Prepare kwargs for chat/generate
        call_kwargs: Dict[str, Any] = {}
        if lora_request is not None:
            call_kwargs["lora_request"] = lora_request
        
        if conversation:
            # Use chat API
            outputs = self._llm.chat([conversation], sp, **call_kwargs)
        else:
            # Use generate API with prompt string
            outputs = self._llm.generate([prompt], sp, **call_kwargs)
        
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
        extras = getattr(cfg, "extra", {}) or {}
        conversation = extras.get("conversation")
        conversations = extras.get("conversations")
        lora_request = self._make_lora_request(cfg)
        
        # Prepare kwargs for chat/generate
        call_kwargs: Dict[str, Any] = {}
        if lora_request is not None:
            call_kwargs["lora_request"] = lora_request
        
        if conversations is not None:
            # Batch with per-prompt conversations
            if len(conversations) != len(prompt_list):
                raise ValueError("conversations length must match prompts length for vllm batch generation.")
            outputs = self._llm.chat(conversations, sp, **call_kwargs)
        elif conversation is not None:
            # Shared conversation for all prompts (not typical, but supported)
            outputs = self._llm.chat([conversation] * len(prompt_list), sp, **call_kwargs)
        else:
            # Standard prompt-based generation
            outputs = self._llm.generate(prompt_list, sp, **call_kwargs)
        
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

