"""
Unit tests for the InferX inference service.
"""

from app.core.bootstrap import bootstrap
from app.models.generation import (
    GenerationRequest,
    GenerationResponse,
)
from app.services.inference_service import (
    InferenceService,
)


def test_generate():

    bootstrap()

    service = InferenceService()

    request = GenerationRequest(
        prompt="Hello InferX",
        max_new_tokens=50,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )

    response = service.generate(request)

    assert isinstance(
        response,
        GenerationResponse,
    )

    assert isinstance(
        response.generated_text,
        str,
    )