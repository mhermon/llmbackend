from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from .base import BaseProvider, CompletedBatch
from .util import require, map_config
from ..types import GenerationConfig, StructuredOutput

Conversation = Sequence[Dict[str, Any]]
ConversationLike = Union[Conversation, Dict[str, Any]]


class MLXProvider(BaseProvider):
    """Local text generation via Apple's MLX (mlx-lm)."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        module = require("mlx_lm", "Install `mlx-lm` to use the 'mlx' provider.")
        load = getattr(module, "load")
        load_kwargs = self.options.get("load_kwargs", {})
        self._model, self._tokenizer = load(path_or_hf_repo=self.model, **load_kwargs)
        self._apply_chat_template = bool(self.options.get("apply_chat_template", False))
        self._add_generation_prompt = self.options.get("add_generation_prompt", True)

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        structured: Optional[StructuredOutput] = None,
    ) -> Any:
        if structured is not None:
            raise ValueError("Structured outputs are only supported for remote providers: openai, anthropic, gemini.")
        module = require("mlx_lm", "Install `mlx-lm`.")
        mlx_generate = getattr(module, "generate")
        cfg = config or GenerationConfig()
        kwargs, conversation, _ = self._generation_kwargs(cfg)
        text_prompt = self._format_prompt(prompt, conversation=conversation)
        text = mlx_generate(self._model, self._tokenizer, text_prompt, **kwargs)
        return text

    def generate_batch(  # type: ignore[override]
        self,
        prompts: Iterable[str],
        config: Optional[GenerationConfig] = None,
        structured: Optional[StructuredOutput] = None,
        batch_options: Optional[Dict[str, Any]] = None,
    ) -> CompletedBatch:
        if structured is not None:
            raise ValueError("Structured outputs are only supported for remote providers: openai, anthropic, gemini.")
        cfg = config or GenerationConfig()
        kwargs, conversation, conversations = self._generation_kwargs(cfg)

        module = require("mlx_lm", "Install `mlx-lm`.")
        batch_generate = getattr(module, "batch_generate", None)
        if batch_generate is None:
            return super().generate_batch(list(prompts), config=cfg, structured=structured, batch_options=batch_options)

        prepared = self._prepare_batch_prompts(list(prompts), conversation, conversations)
        result = batch_generate(self._model, self._tokenizer, prepared, **kwargs)
        texts: Sequence[str]
        if hasattr(result, "texts"):
            texts = list(getattr(result, "texts"))
        elif isinstance(result, (list, tuple)):
            texts = list(result)
        else:
            texts = [getattr(result, "text", str(result))]
        return CompletedBatch(list(texts))

    def _generation_kwargs(
        self, cfg: GenerationConfig
    ) -> tuple[Dict[str, Any], Optional[ConversationLike], Optional[Sequence[ConversationLike]]]:
        kwargs = map_config(
            cfg,
            {
                "max_tokens": "max_tokens",
                "temperature": "temp",
                "top_p": "top_p",
            },
            include_extra=False,
        )
        extras = dict(getattr(cfg, "extra", {}) or {})
        conversation = extras.pop("conversation", None)
        conversations = extras.pop("conversations", None)
        kwargs.update(extras)
        return kwargs, conversation, conversations

    def _prepare_batch_prompts(
        self,
        prompts: List[str],
        shared_conversation: Optional[ConversationLike],
        conversations: Optional[Sequence[ConversationLike]],
    ) -> List[str]:
        if conversations is not None:
            if len(conversations) != len(prompts):
                raise ValueError("conversations length must match prompts length for MLX batch generation.")
            return [
                self._format_prompt(prompts[i], conversation=conversations[i])
                for i in range(len(prompts))
            ]
        return [self._format_prompt(p, conversation=shared_conversation) for p in prompts]

    def _format_prompt(
        self,
        prompt: str,
        conversation: Optional[ConversationLike] = None,
    ) -> str:
        # If a conversation is provided, it fully defines the prompt context
        # and the raw `prompt` string is not used.
        if conversation is not None:
            conv, add_prompt = self._normalize_conversation(conversation)
            return self._tokenizer.apply_chat_template(conv, add_generation_prompt=add_prompt)
        if self._apply_chat_template:
            conv = [{"role": "user", "content": prompt}]
            return self._tokenizer.apply_chat_template(conv, add_generation_prompt=self._add_generation_prompt)
        return prompt

    def _normalize_conversation(
        self, conversation: ConversationLike
    ) -> tuple[Conversation, bool]:
        add_prompt = self._add_generation_prompt
        conv = conversation
        if isinstance(conversation, dict):
            conv = (
                conversation.get("messages")
                or conversation.get("conversation")
                or conversation.get("history")
                or conversation.get("content")
            )
            add_prompt = conversation.get("add_generation_prompt", add_prompt)
        if not isinstance(conv, Sequence) or isinstance(conv, (str, bytes)):
            raise ValueError("Conversation must be a sequence of role/content dicts.")
        return conv, add_prompt

