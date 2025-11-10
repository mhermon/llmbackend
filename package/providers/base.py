from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..types import GenerationConfig, StructuredOutput


class Batch:
    """Provider-agnostic batch handle.

    - OpenAI/Gemini: represents a remote job (id + provider).
    - Anthropic/local: represents a completed local batch with results.
    """

    def __init__(self, provider: "BaseProvider", batch_id: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self._provider = provider
        self._id = batch_id
        self._meta = meta or {}

    @property
    def id(self) -> str:
        return self._id

    def status(self) -> str:
        return self._provider._batch_status(self)

    def results(self) -> List[Any]:
        return self._provider._batch_results(self)

    def __str__(self) -> str:
        return self._id


class CompletedBatch(Batch):
    def __init__(self, results: List[Any]) -> None:
        # Use a sentinel id to indicate completed immediate batches
        super().__init__(provider=DummyProvider(), batch_id="completed", meta={})
        self._results = results

    def status(self) -> str:  # type: ignore[override]
        return "completed"

    def results(self) -> List[Any]:  # type: ignore[override]
        return self._results


class DummyProvider:
    """Minimal stub provider for CompletedBatch."""

    def _batch_status(self, handle: Batch) -> str:  # pragma: no cover - not used
        return "completed"

    def _batch_results(self, handle: Batch) -> List[Any]:  # pragma: no cover - not used
        raise RuntimeError("CompletedBatch should override results()")


class BaseProvider:
    """Abstract text-only provider interface.

    Subclasses must implement `generate`.
    """

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self.options = kwargs

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        structured: Optional[StructuredOutput] = None,
    ) -> Any:  # Usually str; dict/object when structured
        raise NotImplementedError

    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
        structured: Optional[StructuredOutput] = None,
        batch_options: Optional[Dict[str, Any]] = None,
    ) -> Batch:
        """Default sequential per-prompt generation returning a CompletedBatch."""
        results = [self.generate(p, config=config, structured=structured) for p in prompts]
        return CompletedBatch(results)

    # Provider-specific batch operations (override where supported)
    def _batch_status(self, handle: Batch) -> str:
        raise NotImplementedError("Batch status not supported for this provider")

    def _batch_results(self, handle: Batch) -> List[Any]:
        raise NotImplementedError("Batch results not supported for this provider")

    # Helper to construct provider-backed handles
    def _make_handle(self, batch_id: str, meta: Optional[Dict[str, Any]] = None) -> Batch:
        return Batch(self, batch_id, meta or {})
