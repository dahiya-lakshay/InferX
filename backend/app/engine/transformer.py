"""
InferX - Decoder-Only Transformer

Stacks multiple decoder blocks to produce token logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from app.engine.decoder_block import DecoderBlock
from app.engine.embedding import TokenEmbedding
from app.engine.layer_norm import LayerNorm
from app.engine.positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    """
    Decoder-only Transformer model.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()

        self.embedding = TokenEmbedding(
            vocab_size,
            embedding_dim,
        )

        self.position = PositionalEncoding(
            embedding_dim,
        )

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embedding_dim,
                    num_heads,
                    hidden_dim,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = LayerNorm(
            embedding_dim,
        )

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache=None,
        use_cache: bool = False,
    ) -> torch.Tensor:

        """
        Forward pass through the Transformer.

        Returns
        -------
        logits:
            (batch_size, sequence_length, vocab_size)
        """

        x = self.embedding(token_ids)

        x = self.position(x)

        for layer in self.layers:

            x, _ = layer(
                x,
                attention_mask,
            )

        x = self.norm(x)

        logits = self.lm_head(x)

        return logits

