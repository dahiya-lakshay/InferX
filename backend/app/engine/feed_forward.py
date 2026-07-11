"""
InferX - Feed Forward Network

Position-wise Feed Forward Network used inside every
Transformer decoder block.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.linear1 = nn.Linear(
            embedding_dim,
            hidden_dim,
        )

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(
            dropout,
        )

        self.linear2 = nn.Linear(
            hidden_dim,
            embedding_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the feed-forward network.
        """

        x = self.linear1(x)

        x = self.activation(x)

        x = self.dropout(x)

        x = self.linear2(x)

        return x