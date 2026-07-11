"""
InferX - Token Embedding Layer

Converts token IDs into dense vector representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Learns a dense vector representation for every token
    in the vocabulary.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert token IDs into embeddings.

        Parameters
        ----------
        token_ids
            Shape:
                (batch_size, sequence_length)

        Returns
        -------
        Tensor
            Shape:
                (batch_size, sequence_length, embedding_dim)
        """

        return self.embedding(token_ids)