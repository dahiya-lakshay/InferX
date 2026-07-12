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
        description="Input prompt.",
    )

    max_new_tokens: int = Field(
        default=50,
        ge=1,
        le=1024,
    )

    temperature: float = Field(
        default=1.0,
        gt=0,
    )

    top_k: int = Field(
        default=50,
        ge=1,
    )

    top_p: float = Field(
        default=1.0,
        gt=0,
        le=1,
    )

    do_sample: bool = Field(
        default=False,
    )


class GenerationResponse(BaseModel):
    """
    Response returned by the inference API.
    """

    generated_text: str