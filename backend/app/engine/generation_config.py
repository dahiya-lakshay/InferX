"""
InferX - Generation Configuration

Configuration object controlling text generation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GenerationConfig:
    """
    Configuration for autoregressive generation.
    """

    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    do_sample: bool = False
    repetition_penalty: float = 1.0
    eos_token_id: int | None = None
    pad_token_id: int | None = None