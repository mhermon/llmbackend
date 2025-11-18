from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

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
        """Get raw provider-specific status string."""
        return self._provider._batch_status(self)

    def normalized_status(self) -> str:
        """Get normalized status: pending, in_progress, completed, failed, or cancelled."""
        raw = self.status()
        return self._provider._normalize_batch_state(raw)

    def results(self) -> List[Any]:
        """Fetch results from completed batch."""
        return self._provider._batch_results(self)

    def wait(
        self,
        poll_interval: float = 5.0,
        timeout: Optional[float] = None,
        callback: Optional[Callable[[str], None]] = None,
    ) -> "Batch":
        """Block until batch completes, polling at regular intervals.

        Args:
            poll_interval: Seconds between status checks (default: 5.0)
            timeout: Maximum seconds to wait (None = wait forever)
            callback: Optional function called with status on each poll

        Returns:
            Self for chaining (e.g., batch.wait().results())

        Raises:
            RuntimeError: If batch fails or is cancelled
            TimeoutError: If timeout is exceeded
        """
        start = time.time()
        while True:
            normalized = self.normalized_status()
            if callback:
                callback(normalized)

            if normalized == "completed":
                return self
            if normalized in ("failed", "cancelled"):
                raise RuntimeError(f"Batch {self.id} {normalized}: {self.status()}")

            if timeout and (time.time() - start) > timeout:
                raise TimeoutError(
                    f"Batch {self.id} did not complete within {timeout}s (status: {normalized})"
                )

            time.sleep(poll_interval)

    def __str__(self) -> str:
        return self._id


class CompletedBatch(Batch):
    def __init__(self, results: List[Any]) -> None:
        # Use a sentinel id to indicate completed immediate batches
        super().__init__(provider=DummyProvider(), batch_id="completed", meta={})
        self._results = results

    def status(self) -> str:  # type: ignore[override]
        return "completed"

    def normalized_status(self) -> str:  # type: ignore[override]
        return "completed"

    def results(self) -> List[Any]:  # type: ignore[override]
        return self._results

    def wait(  # type: ignore[override]
        self,
        poll_interval: float = 5.0,
        timeout: Optional[float] = None,
        callback: Optional[Callable[[str], None]] = None,
    ) -> "CompletedBatch":
        """CompletedBatch is already done, return immediately."""
        if callback:
            callback("completed")
        return self


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

    def _normalize_batch_state(self, raw_state: str) -> str:
        """Normalize provider-specific batch states to standard values.

        Returns one of: "pending", "in_progress", "completed", "failed", "cancelled"
        """
        state_lower = str(raw_state).lower()

        # Completed states
        if any(x in state_lower for x in ["completed", "succeeded", "success"]):
            return "completed"

        # Failed states
        if any(x in state_lower for x in ["failed", "error"]):
            return "failed"

        # Cancelled states
        if any(x in state_lower for x in ["cancelled", "canceled", "expired"]):
            return "cancelled"

        # In progress states
        if any(x in state_lower for x in ["in_progress", "running", "processing", "finalizing"]):
            return "in_progress"

        # Pending/validating states
        if any(x in state_lower for x in ["pending", "validating", "queued"]):
            return "pending"

        # Default to in_progress for unknown states
        return "in_progress"

    # Helper to construct provider-backed handles
    def _make_handle(self, batch_id: str, meta: Optional[Dict[str, Any]] = None) -> Batch:
        return Batch(self, batch_id, meta or {})

