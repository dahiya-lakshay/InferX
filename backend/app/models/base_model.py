"""
InferX - Base Model Adapter

Defines the interface implemented by every
language model backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseModelAdapter(ABC):
    """
    Base interface for all model backends.
    """

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache=None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Return logits for the supplied input ids.
        """
        raise NotImplementedError