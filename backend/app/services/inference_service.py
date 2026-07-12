"""
InferX - Inference Service

Acts as the bridge between the API layer
and the inference engine.
"""

from __future__ import annotations

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

        # Temporary implementation
        return GenerationResponse(
            generated_text=request.prompt,
        )


inference_service = InferenceService()