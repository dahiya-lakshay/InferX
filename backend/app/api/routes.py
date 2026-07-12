"""
InferX API Routes
"""

from __future__ import annotations

from fastapi import APIRouter

from app.models.generation import (
    GenerationRequest,
    GenerationResponse,
)

router = APIRouter(
    prefix="/api/v1",
    tags=["Inference"],
)


@router.get("/health")
def health():
    """
    Health check endpoint.
    """

    return {
        "status": "healthy",
        "service": "InferX",
    }


@router.post(
    "/generate",
    response_model=GenerationResponse,
)
def generate(
    request: GenerationRequest,
):
    """
    Temporary generation endpoint.
    """

    return GenerationResponse(
        generated_text=request.prompt,
    )