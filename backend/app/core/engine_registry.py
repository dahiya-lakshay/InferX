"""
InferX - Engine Registry

Provides a singleton instance of the inference engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.engine.inference import InferenceEngine


class EngineRegistry:
    """
    Stores the application's inference engine.
    """

    def __init__(self) -> None:

        self._engine: InferenceEngine | None = None

    def register(
        self,
        engine: InferenceEngine,
    ) -> None:
        """
        Register the application's inference engine.
        """

        self._engine = engine

    def get(
        self,
    ) -> InferenceEngine:
        """
        Return the registered inference engine.
        """

        if self._engine is None:
            raise RuntimeError(
                "Inference engine has not been registered."
            )

        return self._engine


engine_registry = EngineRegistry()