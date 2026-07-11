"""
InferX - Scaled Dot-Product Attention

Core attention mechanism from
'Attention Is All You Need'.

This module assumes Query, Key and Value tensors
have already been projected and split into heads.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Computes scaled dot-product attention.

    Expected input shape:

        Query : (B, H, Lq, D)
        Key   : (B, H, Lk, D)
        Value : (B, H, Lv, D)

    where

        B = Batch Size
        H = Number of Heads
        L = Sequence Length
        D = Head Dimension
    """

    def __init__(
        self,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.

        Parameters
        ----------
        query : torch.Tensor
            Shape: (batch_size, num_heads, query_length, head_dim)

        key : torch.Tensor
            Shape: (batch_size, num_heads, key_length, head_dim)

        value : torch.Tensor
            Shape: (batch_size, num_heads, value_length, head_dim)

        mask : torch.Tensor | None
            Optional attention mask.

        Returns
        -------
        tuple
            attention_output :
                (batch_size, num_heads, query_length, head_dim)

            attention_weights :
                (batch_size, num_heads, query_length, key_length)
        """

        if query.dim() != 4:
            raise ValueError("Query must be a 4D tensor.")

        if key.dim() != 4:
            raise ValueError("Key must be a 4D tensor.")

        if value.dim() != 4:
            raise ValueError("Value must be a 4D tensor.")

        batch_size, num_heads, query_length, head_dim = query.shape
        _, _, key_length, _ = key.shape

        # Step 1: Compute attention scores
        attention_scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        )

        # Step 2: Scale the scores
        attention_scores = attention_scores / math.sqrt(head_dim)

        # Step 3: Apply attention mask
        if mask is not None:
            attention_scores = attention_scores.masked_fill(
                mask == 0,
                float("-inf"),
            )

        # Step 4: Convert scores into probabilities
        attention_weights = F.softmax(
            attention_scores,
            dim=-1,
        )

        # Step 5: Apply dropout
        attention_weights = self.dropout(
            attention_weights,
        )

        # Step 6: Weighted sum of values
        attention_output = torch.matmul(
            attention_weights,
            value,
        )

        return attention_output, attention_weights