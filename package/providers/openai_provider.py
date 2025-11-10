from __future__ import annotations

import os
import io
import json
from typing import Any, Optional

from .base import BaseProvider, Batch
from .util import require, map_config
from ..types import GenerationConfig, StructuredOutput


class OpenAIProvider(BaseProvider):
    """OpenAI text generation using the Responses API only."""

    def _get_client(self):
        OpenAI = require("openai", "Install `openai` >= 1.0.0 and set OPENAI_API_KEY.").OpenAI
        api_key = self.options.get("api_key") or os.getenv("OPENAI_API_KEY")
        base_url = self.options.get("base_url") or os.getenv("OPENAI_BASE_URL")
        if not api_key and not base_url:
            raise RuntimeError("Provide OPENAI_API_KEY or a base_url for an OpenAI-compatible server.")
        return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    def _params(self, cfg: GenerationConfig) -> dict[str, Any]:
        params: dict[str, Any] = {"model": self.model}
        params.update(
            map_config(
                cfg,
                {
                    "temperature": "temperature",
                    "top_p": "top_p",
                    "max_tokens": "max_output_tokens",
                    "seed": "seed",
                },
                include_extra=False,
            )
        )
        # Forward provider extras except for pseudo-parameters handled elsewhere.
        extras = dict(getattr(cfg, "extra", {}) or {})
        extras.pop("system", None)
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
        system = cfg.extra.get("system") if cfg else None
        if system:
            params["input"] = [
                {"role": "system", "content": str(system)},
                {"role": "user", "content": str(prompt)},
            ]
        else:
            params["input"] = prompt

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

        resp = client.responses.create(**params)
        text = resp.output_text
        return structured.parse(text) if structured is not None else text

    def generate_batch(
        self,
        prompts: list[str],
        config: Optional[GenerationConfig] = None,
        structured: Optional[StructuredOutput] = None,
        batch_options: Optional[dict[str, Any]] = None,
    ) -> Batch:
        """Create an OpenAI Responses Batch job and return its id.

        batch_options:
            - display_name: str (metadata)
            - completion_window: str (e.g., '24h')
            - custom_ids: list[str] to align outputs
            - system: optional system prompt string
        """
        client = self._get_client()
        cfg = config or GenerationConfig()
        opts = batch_options or {}
        system = opts.get("system") or (cfg.extra.get("system") if cfg else None)
        custom_ids = opts.get("custom_ids")

        lines: list[str] = []
        schema = structured.json_schema() if structured is not None else None
        base_params = self._params(cfg)

        for i, p in enumerate(prompts):
            body: dict[str, Any] = {
                **base_params,
                "model": self.model,
            }
            if system:
                body["input"] = [
                    {"role": "system", "content": str(system)},
                    {"role": "user", "content": str(p)},
                ]
            else:
                body["input"] = str(p)

            if schema is not None:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": structured.name,
                        "schema": schema,
                        "strict": structured.strict,
                    },
                }

            custom_id = custom_ids[i] if isinstance(custom_ids, list) and i < len(custom_ids) else str(i)
            line = json.dumps(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body,
                },
                ensure_ascii=False,
            )
            lines.append(line)

        content = "\n".join(lines).encode("utf-8")
        file_obj = client.files.create(file=io.BytesIO(content), purpose="batch")
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/responses",
            completion_window=opts.get("completion_window", "24h"),
            metadata={"display_name": opts.get("display_name", "batch")},
        )
        handle = self._make_handle(batch.id, meta={
            "custom_ids": custom_ids if isinstance(custom_ids, list) else [str(i) for i in range(len(prompts))],
            "structured": structured,
        })
        return handle

    # Batch status/results
    def _batch_status(self, handle: Batch) -> str:  # type: ignore[override]
        client = self._get_client()
        b = client.batches.retrieve(handle.id)
        return getattr(b, "status", "unknown")

    def _batch_results(self, handle: Batch):  # type: ignore[override]
        client = self._get_client()
        b = client.batches.retrieve(handle.id)
        status = getattr(b, "status", None)
        if status != "completed":
            raise RuntimeError(f"Batch {handle.id} not completed (status={status})")
        ofid = getattr(b, "output_file_id", None)
        if not ofid:
            raise RuntimeError("No output_file_id on completed batch")

        stream = client.files.content(ofid)
        try:
            raw = stream.read()
        except Exception:
            # Fallback to iter_bytes
            chunks = []
            for ch in getattr(stream, "iter_bytes", lambda: [])():
                chunks.append(ch)
            raw = b"".join(chunks)
        text = raw.decode("utf-8")

        results_by_id: dict[str, Any] = {}
        schema = handle._meta.get("structured")  # type: ignore[attr-defined]

        def collect_text(obj: Any) -> str:
            pieces: list[str] = []
            def visit(x: Any):
                if isinstance(x, dict):
                    if isinstance(x.get("text"), str):
                        pieces.append(x["text"])  # type: ignore[index]
                    for v in x.values():
                        visit(v)
                elif isinstance(x, list):
                    for v in x:
                        visit(v)
            visit(obj)
            return "".join(pieces)

        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cid = obj.get("custom_id") or obj.get("id")
            if "error" in obj and obj["error"]:
                results_by_id[str(cid)] = None
                continue
            body = obj.get("response", {}).get("body") or obj.get("response", {})
            if not isinstance(body, dict):
                results_by_id[str(cid)] = None
                continue
            # Extract text
            txt = body.get("output_text")
            if not txt:
                if "output" in body:
                    txt = collect_text(body["output"])  # type: ignore[index]
                elif "choices" in body:
                    try:
                        txt = body["choices"][0]["message"]["content"]
                    except Exception:
                        txt = None
            if txt is None:
                results_by_id[str(cid)] = None
                continue
            if schema is not None:
                try:
                    val = schema.parse(txt)
                except Exception:
                    val = None
            else:
                val = txt
            results_by_id[str(cid)] = val

        order: list[str] = handle._meta.get("custom_ids")  # type: ignore[attr-defined]
        if not order:
            # Fallback to natural ordering by numeric id
            order = sorted(results_by_id.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
        return [results_by_id.get(cid) for cid in order]
