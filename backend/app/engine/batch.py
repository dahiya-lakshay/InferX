"""
InferX - Batch

Represents a batch of active sequences.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from app.engine.sequence import Sequence


@dataclass
class Batch:
    """
    Batch of active sequences.
    """

    sequences: list[Sequence]

    @property
    def batch_size(
        self,
    ) -> int:

        return len(self.sequences)

    @property
    def max_sequence_length(
        self,
    ) -> int:

        if not self.sequences:
            return 0

        return max(
            len(sequence.all_tokens)
            for sequence in self.sequences
        )

    def build_inputs(
        self,
        pad_token_id: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build padded input ids and attention mask.
        """

        max_len = self.max_sequence_length

        input_batch = []
        attention_batch = []

        for sequence in self.sequences:

            tokens = sequence.all_tokens

            padding = max_len - len(tokens)

            input_batch.append(
                tokens + [pad_token_id] * padding
            )

            attention_batch.append(
                [1] * len(tokens)
                + [0] * padding
            )

        return (
            torch.tensor(
                input_batch,
                dtype=torch.long,
            ),
            torch.tensor(
                attention_batch,
                dtype=torch.long,
            ),
        )