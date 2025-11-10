from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from .base import BaseProvider, CompletedBatch
from .util import require, map_config, derive_do_sample
from ..types import GenerationConfig, StructuredOutput

Conversation = Sequence[Dict[str, Any]]
ConversationLike = Union[Conversation, Dict[str, Any]]


class TransformersProvider(BaseProvider):
    """Local text generation via Hugging Face Transformers pipeline."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        transformers = require("transformers", "Install `transformers` to use the 'transformers' provider.")
        pipeline_fn = getattr(transformers, "pipeline")
        pipeline_kwargs: Dict[str, Any] = {"model": self.model}
        pipeline_kwargs.update(self.options.get("pipeline_kwargs", {}))
        model_kwargs = self.options.get("model_kwargs")
        if model_kwargs is not None:
            pipeline_kwargs.setdefault("model_kwargs", model_kwargs)
        tokenizer_ref = self.options.get("tokenizer")
        if tokenizer_ref and "tokenizer" not in pipeline_kwargs:
            pipeline_kwargs["tokenizer"] = tokenizer_ref
        self._pipeline = pipeline_fn("text-generation", **pipeline_kwargs)
        self._tokenizer = getattr(self._pipeline, "tokenizer", None)
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
        cfg = config or GenerationConfig()
        kwargs, conversation, _ = self._generation_kwargs(cfg)
        text_prompt = self._format_prompt(prompt, conversation=conversation)
        out = self._pipeline(text_prompt, **kwargs)
        return self._first_text(out)

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
        prompt_list = list(prompts)
        if not prompt_list:
            return CompletedBatch([])
        kwargs, shared_conversation, conversations = self._generation_kwargs(cfg)
        prepared = self._prepare_batch_prompts(prompt_list, shared_conversation, conversations)
        out = self._pipeline(prepared, **kwargs)
        texts = [self._first_text(item) for item in out]
        return CompletedBatch(texts)

    def _base_kwargs(self, cfg: GenerationConfig) -> Dict[str, Any]:
        d: dict[str, Any] = {"return_full_text": False, "do_sample": derive_do_sample(cfg)}
        d.update(
            map_config(
                cfg,
                {
                    "max_tokens": "max_new_tokens",
                    "temperature": "temperature",
                    "top_p": "top_p",
                },
                include_extra=False,
            )
        )
        return d

    def _generation_kwargs(
        self, cfg: GenerationConfig
    ) -> tuple[Dict[str, Any], Optional[ConversationLike], Optional[Sequence[ConversationLike]]]:
        kwargs = self._base_kwargs(cfg)
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
                raise ValueError("conversations length must match prompts length for transformers batch generation.")
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
            tokenizer = self._require_tokenizer("conversation input")
            return tokenizer.apply_chat_template(conv, add_generation_prompt=add_prompt)
        if self._apply_chat_template:
            tokenizer = self._require_tokenizer("apply_chat_template")
            conv = [{"role": "user", "content": prompt}]
            return tokenizer.apply_chat_template(conv, add_generation_prompt=self._add_generation_prompt)
        return prompt

    def _first_text(self, item: Any) -> str:
        if isinstance(item, list) and item:
            entry = item[0]
            return entry.get("generated_text", "")
        if isinstance(item, dict):
            return item.get("generated_text", "")
        return ""

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

    def _require_tokenizer(self, context: str):
        if self._tokenizer is None:
            raise RuntimeError(f"Transformers pipeline has no tokenizer; cannot process {context}.")
        return self._tokenizer

