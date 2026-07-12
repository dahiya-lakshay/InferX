"""
InferX - Sampler

Selects the next token according to the
generation configuration.
"""

from __future__ import annotations

import torch

from app.engine.generation_config import GenerationConfig
from app.engine.sampling import (
    greedy_sample,
    random_sample,
    top_k_sample,
    top_p_sample,
)


class Sampler:
    """
    High-level sampling interface.
    """

    def sample(
      self,
        logits: torch.Tensor,
        config: GenerationConfig,
    ) -> int:
        """
        Sample the next token.
        """

        if not config.do_sample:
            return int(greedy_sample(logits).item())

        if config.top_k > 0:
            return int(
                top_k_sample(
                    logits,
                    config.top_k,
                ).item()
            )

        if config.top_p < 1.0:
            return int(
                top_p_sample(
                    logits,
                    config.top_p,
                ).item()
            )

        return int(random_sample(logits).item())