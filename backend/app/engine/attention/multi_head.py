"""
InferX - Multi-Head Self-Attention

Implements multi-head self-attention using the
ScaledDotProductAttention module.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from app.engine.attention.scaled_attention import (
    ScaledDotProductAttention,
)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.key_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.value_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.attention = ScaledDotProductAttention(
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-head self-attention.

        Parameters
        ----------
        x
            Shape:
                (batch_size, sequence_length, embedding_dim)

        mask
            Optional attention mask.

        Returns
        -------
        tuple
            output :
                (batch_size, sequence_length, embedding_dim)

            attention_weights :
                (batch_size, num_heads, sequence_length, sequence_length)
        """

        batch_size, sequence_length, _ = x.shape

        # Step 1: Project input into Q, K and V

        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        # Step 2: Split into multiple heads

        query = query.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        key = key.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        value = value.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # Step 3: Compute attention

        attention_output, attention_weights = self.attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
        )

        # Step 4: Merge attention heads

        attention_output = attention_output.transpose(
            1,
            2,
        ).contiguous()

        attention_output = attention_output.view(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )

        # Step 5: Final output projection

        output = self.output_projection(
            attention_output,
        )

        return output, attention_weights