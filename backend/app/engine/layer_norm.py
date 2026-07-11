"""
InferX - Layer Normalization

Implementation of Layer Normalization used in Transformer
architectures.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Custom implementation of Layer Normalization.
    """

    def __init__(
        self,
        embedding_dim: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.eps = eps

        self.gamma = nn.Parameter(
            torch.ones(embedding_dim)
        )

        self.beta = nn.Parameter(
            torch.zeros(embedding_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalize over the last dimension.
        """

        mean = x.mean(
            dim=-1,
            keepdim=True,
        )

        variance = x.var(
            dim=-1,
            unbiased=False,
            keepdim=True,
        )

        normalized = (
            x - mean
        ) / torch.sqrt(
            variance + self.eps
        )

        return (
            self.gamma * normalized
            + self.beta
        )