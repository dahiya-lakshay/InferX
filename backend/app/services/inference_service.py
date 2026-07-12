"""
InferX - Inference Service

Acts as the bridge between the API layer
and the inference engine.
"""

from __future__ import annotations

from app.core.engine_registry import engine_registry
from app.engine.generation_config import GenerationConfig
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

        engine = engine_registry.get()

        config = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )

        generated_text = engine.generate(
            prompt=request.prompt,
            config=config,
        )

        return GenerationResponse(
            generated_text=generated_text,
        )


inference_service = InferenceService()