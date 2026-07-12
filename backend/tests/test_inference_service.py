"""
Unit tests for the InferX inference service.
"""

from app.models.generation import (
    GenerationRequest,
    GenerationResponse,
)
from app.services.inference_service import (
    InferenceService,
)


def test_generate():

    service = InferenceService()

    request = GenerationRequest(
        prompt="Hello InferX",
        max_new_tokens=50,
    )

    response = service.generate(request)

    assert isinstance(
        response,
        GenerationResponse,
    )

    assert (
        response.generated_text
        == "Hello InferX"
    )