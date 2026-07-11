"""
InferX - Decode Step

Executes one decoding iteration for an entire batch.
"""

from __future__ import annotations

import torch

from app.engine.batch import Batch
from app.engine.sampling import greedy_sample


class DecodeStep:
    """
    Executes one decoding step for every
    active sequence.
    """

    def __init__(self, model) -> None:

        self.model = model

    def forward(
        self,
        batch: Batch,
    ) -> None:
        """
        Generate one token for every sequence.
        """

        input_ids = batch.build_input_ids()

        logits = self.model(input_ids)

        next_token_logits = logits[:, -1, :]

        next_tokens = greedy_sample(
            next_token_logits,
        )

        for sequence, token in zip(
            batch.sequences,
            next_tokens,
        ):
            sequence.append_token(
                int(token.item())
            )