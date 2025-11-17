from __future__ import annotations

import os
import warnings
from typing import Any, Optional

from .base import BaseProvider
from .util import require, map_config, parse_structured_output
from ..types import GenerationConfig, StructuredOutput


class GeminiProvider(BaseProvider):
    """Google Gemini via google-generativeai, optional JSON schema."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self._genai = None
        self._model = None
        self._client = None

    def _api_key(self) -> str:
        api_key = self.options.get("api_key") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Provide GOOGLE_API_KEY or pass api_key.")
        return api_key

    def _genai_module(self):
        if self._genai is None:
            self._genai = require(
                "google.generativeai", "Install `google-generativeai` and set GOOGLE_API_KEY."
            )
        return self._genai

    def _get_model(self):
        if self._model is None:
            genai = self._genai_module()
            genai.configure(api_key=self._api_key())
            self._model = genai.GenerativeModel(self.model)
        return self._model

    def _get_client(self):
        if self._client is None:
            genai = self._genai_module()
            self._client = genai.Client(api_key=self._api_key())
        return self._client

    def _gen_cfg(self, cfg: GenerationConfig) -> dict[str, Any]:
        params = map_config(
            cfg,
            {
                "temperature": "temperature",
                "top_p": "top_p",
                "max_tokens": "max_output_tokens",
            },
            include_extra=False,
        )
        extras = dict(getattr(cfg, "extra", {}) or {})
        extras.pop("system_instruction", None)
        params.update(extras)
        return params

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        structured: Optional[StructuredOutput] = None,
    ) -> Any:
        model = self._get_model()
        cfg = config or GenerationConfig()

        kwargs: dict[str, Any] = {"generation_config": self._gen_cfg(cfg)}
        system_instruction = cfg.extra.get("system_instruction")
        if system_instruction:
            kwargs["system_instruction"] = {"parts": [{"text": str(system_instruction)}]}
        if structured is not None:
            kwargs["response_mime_type"] = "application/json"
            schema = structured.json_schema()
            if schema is not None:
                kwargs["response_schema"] = schema

        resp = model.generate_content(prompt, **kwargs)
        text = resp.text or ""
        if structured is not None:
            return parse_structured_output(structured, text, "gemini", "response")
        return text

    def generate_batch(
        self,
        prompts: list[str],
        config: Optional[GenerationConfig] = None,
        structured: Optional[StructuredOutput] = None,
        batch_options: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Create a Gemini batch job and return its name/ID."""
        client = self._get_client()

        cfg = config or GenerationConfig()
        opts = batch_options or {}
        system_instruction = cfg.extra.get("system_instruction") if cfg else None
        raw_custom_ids = opts.get("custom_ids")

        inline_requests: list[dict[str, Any]] = []
        schema = structured.json_schema() if structured is not None else None
        gen_cfg = self._gen_cfg(cfg)
        resolved_ids: list[str] = []
        for idx, p in enumerate(prompts):
            if isinstance(raw_custom_ids, list) and idx < len(raw_custom_ids):
                cid = str(raw_custom_ids[idx])
            else:
                cid = str(idx)
            resolved_ids.append(cid)
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
        return self._make_handle(job.name, meta={"structured": structured, "custom_ids": resolved_ids})

    def _batch_status(self, handle):  # type: ignore[override]
        client = self._get_client()
        job = client.batches.get(handle.id)
        state = getattr(job, "state", None)
        return getattr(state, "name", state) or getattr(job, "status", "unknown")

    def _batch_results(self, handle):  # type: ignore[override]
        client = self._get_client()
        job = client.batches.get(handle.id)
        state = getattr(job, "state", None)
        state_name = getattr(state, "name", state)
        completed = {
            "SUCCEEDED",
            "COMPLETED",
            "completed",
            "JOB_STATE_SUCCEEDED",
        }
        if state_name not in completed:
            raise RuntimeError(f"Batch {handle.id} not completed (state={state_name})")

        structured_cfg = handle._meta.get("structured")  # type: ignore[attr-defined]
        expected_ids: list[str] = handle._meta.get("custom_ids") or []  # type: ignore[attr-defined]

        def collect_text(obj: Any) -> str:
            out: list[str] = []
            stack = [obj]
            while stack:
                current = stack.pop()
                if isinstance(current, dict):
                    text_val = current.get("text")
                    if isinstance(text_val, str):
                        out.append(text_val)
                    stack.extend(current.values())
                elif isinstance(current, list):
                    stack.extend(current)
            return "".join(out)

        dest = getattr(job, "dest", None)
        inline_responses = getattr(dest, "inlined_responses", None) if dest else None
        outputs_by_id: dict[str, Optional[str]] = {}

        if inline_responses:
            for idx, entry in enumerate(inline_responses):
                cid = expected_ids[idx] if idx < len(expected_ids) else str(idx)
                error = getattr(entry, "error", None)
                if error:
                    warnings.warn(f"Gemini batch {handle.id} error for {cid}: {error}")
                    outputs_by_id[cid] = None
                    continue
                outputs_by_id[cid] = self._inline_response_text(entry)
        else:
            payload = (
                getattr(job, "result", None)
                or getattr(job, "results", None)
                or getattr(job, "outputs", None)
            )
            if payload is None:
                raise RuntimeError(
                    "Could not locate Gemini batch results. Inspect the job via client.batches.get(name)."
                )
            items = payload if isinstance(payload, list) else [payload]
            for idx, item in enumerate(items):
                cid = expected_ids[idx] if idx < len(expected_ids) else str(idx)
                outputs_by_id[cid] = collect_text(item)

        ordered_ids = expected_ids if expected_ids else sorted(outputs_by_id.keys())
        texts = [outputs_by_id.get(cid) for cid in ordered_ids]

        if structured_cfg is not None:
            return [
                parse_structured_output(
                    structured_cfg,
                    text,
                    "gemini",
                    f"batch:{handle.id}:{cid}",
                    on_error="warn",
                )
                for cid, text in zip(ordered_ids, texts)
            ]
        return texts

    def _inline_response_text(self, entry: Any) -> Optional[str]:
        """Extract best-effort text from an inline Gemini response entry."""
        resp_obj = getattr(entry, "response", None)
        if resp_obj is None:
            return None
        text_attr = getattr(resp_obj, "text", None)
        if isinstance(text_attr, str) and text_attr:
            return text_attr
        to_dict = getattr(resp_obj, "to_dict", None)
        if callable(to_dict):
            data = to_dict()
            if isinstance(data, (dict, list)):
                txt = self._collect_text(data)
                if txt:
                    return txt
        candidates = getattr(resp_obj, "candidates", None)
        if candidates:
            return self._collect_text(candidates)
        return str(resp_obj)

    def _collect_text(self, obj: Any) -> str:
        out: list[str] = []
        stack = [obj]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                text_val = current.get("text")
                if isinstance(text_val, str):
                    out.append(text_val)
                stack.extend(current.values())
            elif isinstance(current, list):
                stack.extend(current)
        return "".join(out)
