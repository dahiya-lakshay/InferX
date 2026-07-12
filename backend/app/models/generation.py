"""
InferX API Models
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    """
    Request body for text generation.
    """

    prompt: str = Field(
        ...,
        description="Input prompt for generation.",
    )

    max_new_tokens: int = Field(
        default=50,
        ge=1,
        le=1024,
        description="Maximum number of new tokens to generate.",
    )


class GenerationResponse(BaseModel):
    """
    Response returned by the inference API.
    """

    generated_text: str