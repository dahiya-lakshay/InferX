"""
InferX API Routes
"""

from __future__ import annotations

from fastapi import APIRouter

from app.models.generation import (
    GenerationRequest,
    GenerationResponse,
)

from app.services.inference_service import inference_service

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
    Generate text using the inference service.
    """

    return inference_service.generate(request)