"""
InferX - Transformer Model Adapter

Wraps the native InferX Transformer behind the
BaseModelAdapter interface.
"""

from __future__ import annotations

import torch

from app.engine.transformer import Transformer
from app.models.base_model import BaseModelAdapter


class TransformerAdapter(BaseModelAdapter):
    """
    Adapter for the native InferX Transformer.
    """

    def __init__(
        self,
        model: Transformer,
    ) -> None:

        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache=None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the wrapped transformer.
        """

        return self.model(
            token_ids=input_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )