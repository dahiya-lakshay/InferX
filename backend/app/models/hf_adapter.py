"""
InferX - Hugging Face Model Adapter

Wraps Hugging Face causal language models behind the
InferX BaseModelAdapter interface.
"""

from __future__ import annotations

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from app.models.base_model import BaseModelAdapter


class HFAdapter(BaseModelAdapter):
    """
    Hugging Face backend implementing the
    BaseModelAdapter interface.
    """

    def __init__(
        self,
        model_name: str = "sshleifer/tiny-gpt2",
    ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
        )

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache=None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the Hugging Face model.

        Returns logits of shape:
        (batch_size, sequence_length, vocab_size)
        """

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict=True,
        )

        return outputs.logits