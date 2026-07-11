"""
InferX - Key Value Cache

Stores past attention keys and values for efficient
autoregressive decoding.
"""

from __future__ import annotations

import torch


class KVCache:
    """
    Cache previously computed Keys and Values.
    """

    def __init__(self) -> None:

        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None

    def clear(self) -> None:
        """
        Reset cache.
        """

        self.keys = None
        self.values = None

    def update(
        self,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append new Keys and Values to the cache.
        """

        if self.keys is None:

            self.keys = new_keys

            self.values = new_values

        else:

            self.keys = torch.cat(
                [
                    self.keys,
                    new_keys,
                ],
                dim=2,
            )

            self.values = torch.cat(
                [
                    self.values,
                    new_values,
                ],
                dim=2,
            )

        return self.keys, self.values

    def get(
        self,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """
        Return cached tensors.
        """

        return self.keys, self.values