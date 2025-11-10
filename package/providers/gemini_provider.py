from __future__ import annotations

import os
from typing import Any, Optional

from .base import BaseProvider
from .util import require, map_config
from ..types import GenerationConfig, StructuredOutput


class GeminiProvider(BaseProvider):
    """Google Gemini via google-generativeai, optional JSON schema."""

    def _get_model(self):
        genai = require("google.generativeai", "Install `google-generativeai` and set GOOGLE_API_KEY.")
        api_key = self.options.get("api_key") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Provide GOOGLE_API_KEY or pass api_key.")
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(self.model)

    def _gen_cfg(self, cfg: GenerationConfig) -> dict[str, Any]:
        return map_config(cfg, {
            "temperature": "temperature",
            "top_p": "top_p",
            "max_tokens": "max_output_tokens",
        })

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        structured: Optional[StructuredOutput] = None,
    ) -> Any:
        model = self._get_model()
        cfg = config or GenerationConfig()

        kwargs: dict[str, Any] = {"generation_config": self._gen_cfg(cfg)}
        if structured is not None:
            kwargs["response_mime_type"] = "application/json"
            schema = structured.json_schema()
            if schema is not None:
                kwargs["response_schema"] = schema

        resp = model.generate_content(prompt, **kwargs)
        text = resp.text or ""
        return structured.parse(text) if structured is not None else text

    def generate_batch(
        self,
        prompts: list[str],
        config: Optional[GenerationConfig] = None,
        structured: Optional[StructuredOutput] = None,
        batch_options: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Create a Gemini batch job and return its name/ID."""
        genai = require("google.generativeai", "Install `google-generativeai` and set GOOGLE_API_KEY.")
        api_key = self.options.get("api_key") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Provide GOOGLE_API_KEY or pass api_key.")
        client = genai.Client(api_key=api_key)

        cfg = config or GenerationConfig()
        opts = batch_options or {}
        system_instruction = opts.get("system_instruction") or (cfg.extra.get("system_instruction") if cfg else None)

        inline_requests: list[dict[str, Any]] = []
        schema = structured.json_schema() if structured is not None else None
        gen_cfg = self._gen_cfg(cfg)
        for p in prompts:
            config_block: dict[str, Any] = {**gen_cfg}
            if structured is not None:
                config_block["response_mime_type"] = "application/json"
                if schema is not None:
                    config_block["response_schema"] = schema
            if system_instruction:
                config_block["system_instruction"] = {"parts": [{"text": str(system_instruction)}]}
            inline_requests.append(
                {
                    "contents": [{"role": "user", "parts": [{"text": str(p)}]}],
                    "config": config_block,
                }
            )

        job = client.batches.create(
            model=self.model,
            src=inline_requests,
            config={"display_name": opts.get("display_name", "batch")},
        )
        # Return a provider-backed handle
        return self._make_handle(job.name, meta={"structured": structured})

    def _batch_status(self, handle):  # type: ignore[override]
        genai = require("google.generativeai", "Install `google-generativeai` and set GOOGLE_API_KEY.")
        api_key = self.options.get("api_key") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Provide GOOGLE_API_KEY or pass api_key.")
        client = genai.Client(api_key=api_key)
        job = client.batches.get(handle.id)
        return getattr(job, "state", getattr(job, "status", "unknown"))

    def _batch_results(self, handle):  # type: ignore[override]
        genai = require("google.generativeai", "Install `google-generativeai` and set GOOGLE_API_KEY.")
        api_key = self.options.get("api_key") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Provide GOOGLE_API_KEY or pass api_key.")
        client = genai.Client(api_key=api_key)
        job = client.batches.get(handle.id)
        state = getattr(job, "state", getattr(job, "status", None))
        if state not in ("SUCCEEDED", "COMPLETED", "completed"):
            raise RuntimeError(f"Batch {handle.id} not completed (state={state})")

        # Best-effort extraction of text responses from job result payloads.
        structured = handle._meta.get("structured")  # type: ignore[attr-defined]

        def collect_text(x):
            out = []
            def walk(v):
                if isinstance(v, dict):
                    # Gemini typical: candidates -> content -> parts -> {text}
                    if "text" in v and isinstance(v["text"], str):
                        out.append(v["text"])
                    for val in v.values():
                        walk(val)
                elif isinstance(v, list):
                    for val in v:
                        walk(val)
            walk(x)
            return "".join(out)

        # Try common fields: result, results, outputs
        payload = getattr(job, "result", None) or getattr(job, "results", None) or getattr(job, "outputs", None)
        if payload is None:
            # As a fallback, expose raw job object to the caller for manual handling.
            raise RuntimeError("Could not locate results on Gemini batch job. Inspect job via client.batches.get(name).")

        items = payload if isinstance(payload, list) else [payload]
        texts = [collect_text(item) for item in items]
        if structured is not None:
            out = []
            for t in texts:
                try:
                    out.append(structured.parse(t))
                except Exception:
                    out.append(None)
            return out
        return texts
