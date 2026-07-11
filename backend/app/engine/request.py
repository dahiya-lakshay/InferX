"""
InferX - Generation Request

Represents a single text generation request.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.engine.generation_config import GenerationConfig


@dataclass(slots=True)
class GenerationRequest:
    """
    Represents a single inference request.
    """

    request_id: str

    prompt: str

    config: GenerationConfig

    arrival_time: float = field(
        default_factory=time.time
    )