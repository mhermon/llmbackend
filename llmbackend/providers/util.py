from __future__ import annotations

import importlib
from typing import Any, Dict, Optional


def require(module: str, hint: str = ""):
    """Import a module or raise a clean ImportError with an optional hint."""
    try:
        return importlib.import_module(module)
    except Exception as e:
        msg = f"Missing dependency: {module}. {hint}".strip()
        raise ImportError(msg) from e


def map_config(cfg, mapping: Dict[str, str], include_extra: bool = True) -> Dict[str, Any]:
    """Map common GenerationConfig fields to provider-specific parameter names.

    - `mapping` maps canonical names (temperature, top_p, max_tokens, seed)
      to provider parameter keys.
    - Ignores None values.
    - Optionally merges `cfg.extra`.
    """
    params: Dict[str, Any] = {}
    for src, dst in mapping.items():
        val = getattr(cfg, src, None)
        if val is not None:
            params[dst] = val
    if include_extra:
        params.update(getattr(cfg, "extra", {}) or {})
    return params


def derive_do_sample(cfg) -> bool:
    """Heuristic for local sampling: sample when temperature > 0 or top_p < 1.

    Keeps generation deterministic by default.
    """
    t = getattr(cfg, "temperature", None)
    p = getattr(cfg, "top_p", None)
    return (t is not None and t > 0) or (p is not None and p < 1.0)


def parse_structured_output(
    structured,  # StructuredOutput, but keep untyped to avoid circular import
    text: Optional[str],
    provider: str,
    source: str,
    *,
    on_error: str = "raise",
) -> Any:
    """Parse structured output with a consistent, contextual error message."""
    if text is None or text == "":
        msg = f"{provider}: empty structured response for {source}"
        if on_error == "warn":
            import warnings  # Local import to keep import-time light
            warnings.warn(msg)
            return None
        raise ValueError(msg)
    try:
        return structured.parse(text)
    except Exception as exc:  # pragma: no cover - depends on provider SDKs
        snippet = text.strip()
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        msg = f"{provider}: failed to parse structured output for {source}: {exc}. Raw: {snippet}"
        if on_error == "warn":
            import warnings  # Local import to keep import-time light
            warnings.warn(msg)
            return None
        raise ValueError(msg) from exc

