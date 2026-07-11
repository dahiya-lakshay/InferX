"""
InferX - Inference Engine

Coordinates tokenization, model inference,
sampling, and decoding.
"""

from __future__ import annotations

import torch

from app.engine.transformer import Transformer
from app.tokenizer.tokenizer import Tokenizer
from app.engine.sampling import greedy_sampling
from app.engine.generation_config import GenerationConfig

class InferenceEngine:
    """
    High-level inference engine for InferX.
    """

    def __init__(
        self,
        model: Transformer,
        tokenizer: Tokenizer,
        sampling_strategy: str = "greedy",
    ) -> None:

        self.model = model
        self.tokenizer = tokenizer
        self.sampling_strategy = sampling_strategy

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> str:

        """
        Generate text autoregressively.
        """

        if config is None:
            config = GenerationConfig()

        # Encode prompt
        token_ids = self.tokenizer.encode(prompt)

        generated_tokens = list(token_ids)

        for _ in range(config.max_new_tokens):

            input_tensor = torch.tensor(
                [generated_tokens],
                dtype=torch.long,
            )

            logits = self._forward(input_tensor)

            next_token = self._sample(logits)

            generated_tokens.append(next_token)

        return self._decode(generated_tokens)


    def _forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        """

        return self.model(token_ids)

    def _sample(
        self,
        logits: torch.Tensor,
    ) -> int:
        """
        Sample the next token from model logits.

        Currently uses greedy decoding.
        """

        next_token_logits = logits[:, -1, :]

        token = greedy_sampling(next_token_logits)

        return int(token.item())

    def _decode(
        self,
        token_ids: list[int],
    ) -> str:
        """
        Decode generated token ids.
        """

        return self.tokenizer.decode(token_ids)