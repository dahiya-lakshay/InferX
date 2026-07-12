"""
InferX - Decode Step

Executes one decoding iteration for an entire batch.
"""

from __future__ import annotations

from app.engine.batch import Batch
from app.engine.generation_config import GenerationConfig
from app.engine.sampler import Sampler


class DecodeStep:
    """
    Executes one decoding step for every
    active sequence.
    """

    def __init__(
        self,
        model,
    ) -> None:

        self.model = model
        self.sampler = Sampler()

    def forward(
        self,
        batch: Batch,
    ) -> None:
        """
        Generate one token for every sequence.
        """

        input_ids = batch.build_input_ids()

        logits = self.model.forward(
            input_ids=input_ids,
        )

        next_token_logits = logits[:, -1, :]

        for sequence, token_logits in zip(
            batch.sequences,
            next_token_logits,
        ):

            next_token = self.sampler.sample(
                token_logits,
                sequence.request.config,
            )

            sequence.append_token(next_token)