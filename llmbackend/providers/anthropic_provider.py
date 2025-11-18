from __future__ import annotations

import os
from typing import Any, Optional

from .base import BaseProvider
from .util import require, map_config
from ..types import GenerationConfig, StructuredOutput


class AnthropicProvider(BaseProvider):
    """Anthropic Claude via Messages API, with optional JSON schema."""

    def _get_client(self):
        Anthropic = require("anthropic", "Install `anthropic` and set ANTHROPIC_API_KEY.").Anthropic
        api_key = self.options.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Provide ANTHROPIC_API_KEY or pass api_key.")
        return Anthropic(api_key=api_key)

    def _params(self, cfg: GenerationConfig) -> dict[str, Any]:
        params: dict[str, Any] = {"model": self.model}
        params.update(
            map_config(
                cfg,
                {
                    "max_tokens": "max_tokens",
                    "temperature": "temperature",
                    "top_p": "top_p",
                },
                include_extra=False,
            )
        )
        extras = dict(getattr(cfg, "extra", {}) or {})
        extras.pop("system", None)
        extras.pop("system_instruction", None)
        params.update(extras)
        return params

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        structured: Optional[StructuredOutput] = None,
    ) -> Any:
        client = self._get_client()
        cfg = config or GenerationConfig()
        params = self._params(cfg)
        params["messages"] = [{"role": "user", "content": prompt}]
        system_prompt = cfg.extra.get("system") or cfg.extra.get("system_instruction")
        if system_prompt:
            params["system"] = str(system_prompt)

        if structured is not None:
            schema = structured.json_schema()
            if schema is not None:
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": structured.name,
                        "schema": schema,
                        "strict": structured.strict,
                    },
                }

        resp = client.messages.create(**params)
        text = "".join([b.text for b in resp.content if getattr(b, "type", "text") == "text"])  # type: ignore
        return structured.parse(text) if structured is not None else text
