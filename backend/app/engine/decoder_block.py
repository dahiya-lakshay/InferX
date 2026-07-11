"""
InferX - Transformer Decoder Block

Implements a Pre-LayerNorm Transformer decoder block.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from app.engine.attention.multi_head import MultiHeadAttention
from app.engine.feed_forward import FeedForward
from app.engine.layer_norm import LayerNorm


class DecoderBlock(nn.Module):
    """
    Single Transformer Decoder Block.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1 = LayerNorm(embedding_dim)

        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.norm2 = LayerNorm(embedding_dim)

        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache=None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the decoder block.
        """

        residual = x

        x = self.norm1(x)

        attention_output, attention_weights = self.attention(
            x=x,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )

        x = residual + self.dropout(attention_output)

        residual = x

        x = self.norm2(x)

        feed_forward_output = self.feed_forward(x)

        x = residual + self.dropout(feed_forward_output)

        return x, attention_weights