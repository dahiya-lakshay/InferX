"""
InferX - Inference Service

Acts as the bridge between the API layer
and the inference engine.
"""

from __future__ import annotations

from app.core.engine_registry import engine_registry
from app.models.generation import (
    GenerationRequest,
    GenerationResponse,
)


class InferenceService:
    """
    Handles inference requests.
    """

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """
        Process a generation request.
        """

        # Retrieve the application's inference engine.
        # The actual generation call will be integrated
        # in the next phase.
        engine = engine_registry.get()

        # Prevent unused-variable warnings until the
        # engine is connected.
        _ = engine

        return GenerationResponse(
            generated_text=request.prompt,
        )


inference_service = InferenceService()