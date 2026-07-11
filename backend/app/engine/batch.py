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

    def build_input_ids(
        self,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """
        Build padded input tensor.
        """

        max_len = self.max_sequence_length

        batch = []

        for sequence in self.sequences:

            tokens = sequence.all_tokens

            padded = tokens + [
                pad_token_id
            ] * (max_len - len(tokens))

            batch.append(padded)

        return torch.tensor(
            batch,
            dtype=torch.long,
        )