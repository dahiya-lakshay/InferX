"""
InferX - Sinusoidal Positional Encoding

Injects positional information into token embeddings.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding from
    'Attention Is All You Need'.
    """

    def __init__(
        self,
        embedding_dim: int,
        max_length: int = 5000,
    ) -> None:
        super().__init__()

        pe = torch.zeros(max_length, embedding_dim)

        position = torch.arange(
            0,
            max_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                embedding_dim,
                2,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add positional encoding to embeddings.

        Input shape:
            (batch_size, sequence_length, embedding_dim)

        Output shape:
            (batch_size, sequence_length, embedding_dim)
        """

        sequence_length = embeddings.size(1)

        return embeddings + self.pe[:, :sequence_length]