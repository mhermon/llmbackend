from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Iterable, Type


@dataclass
class GenerationConfig:
    """Common generation parameters across providers.

    Only include fields you actually care about. Unknown/None fields are
    ignored by providers.
    """

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    # Extra provider-specific params (e.g., presence_penalty, stop)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_maybe_dict(cls, cfg: Optional[Dict[str, Any] | "GenerationConfig"]) -> "GenerationConfig":
        if cfg is None:
            return cls()
        if isinstance(cfg, GenerationConfig):
            return cfg
        if not isinstance(cfg, dict):
            raise TypeError("config must be dict or GenerationConfig")
        canonical = ("temperature", "top_p", "max_tokens", "seed")
        base = {k: cfg.get(k) for k in canonical}

        merged_extra: Dict[str, Any] = {}
        inline_extra = cfg.get("extra")
        if isinstance(inline_extra, dict):
            merged_extra.update(inline_extra)
        for key, value in cfg.items():
            if key not in canonical and key != "extra":
                merged_extra[key] = value

        return cls(**base, extra=merged_extra)


JsonDict = Dict[str, Any]


def _maybe_pydantic_base():
    try:
        from pydantic import BaseModel  # type: ignore
        return BaseModel
    except Exception:
        return None


BaseModel = _maybe_pydantic_base()


@dataclass
class StructuredOutput:
    """Structured output configuration.

    Provide either a JSON schema dict via `schema`, or a Pydantic model via
    `pydantic_model`. If both are given, `schema` takes precedence. Structured
    outputs are supported only by remote providers (OpenAI, Anthropic, Gemini).

    - `name` is used where providers require a schema name.
    - `parser` can post-process the parsed JSON/object.
    """

    schema: Optional[JsonDict] = None
    pydantic_model: Optional[Type[Any]] = None
    name: str = "Response"
    strict: bool = True
    parser: Optional[Callable[[JsonDict], Any]] = None

    def json_schema(self) -> Optional[JsonDict]:
        if self.schema is not None:
            return self.schema
        if self.pydantic_model is not None:
            # Pydantic v2: model_json_schema; v1: schema
            model = self.pydantic_model
            try:
                # v2
                return model.model_json_schema()  # type: ignore[attr-defined]
            except Exception:
                try:
                    # v1
                    return model.schema()  # type: ignore[attr-defined]
                except Exception:
                    pass
        return None

    def parse(self, text: str) -> Any:
        import json
        # Prefer Pydantic parsing when a model is provided
        if self.pydantic_model is not None:
            try:
                # v2
                return self.pydantic_model.model_validate_json(text)  # type: ignore[attr-defined]
            except Exception:
                try:
                    # v1
                    return self.pydantic_model.parse_raw(text)  # type: ignore[attr-defined]
                except Exception:
                    pass

        data = json.loads(text)
        if self.parser is not None:
            return self.parser(data)
        return data

