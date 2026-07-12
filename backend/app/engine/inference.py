"""
InferX - Inference Engine

Coordinates tokenization, model inference,
sampling, and decoding.
"""

from __future__ import annotations

import torch

from app.models.base_model import BaseModelAdapter
from app.tokenizer.tokenizer import Tokenizer
from app.engine.generation_config import GenerationConfig
from app.engine.stopping import StoppingCriteria
from app.engine.sequence import Sequence
from app.engine.request import GenerationRequest
from app.engine.sampler import Sampler

class InferenceEngine:
    """
    High-level inference engine for InferX.
    """

    def __init__(
        self,
        model: BaseModelAdapter,
        tokenizer: Tokenizer,
    ) -> None:

        self.model = model
        self.tokenizer = tokenizer
        self.stopping = StoppingCriteria()
        self.sampler = Sampler()

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

        request = GenerationRequest(
            request_id="request_0",
            prompt=prompt,
            config=config,
        )

        sequence = Sequence(request=request)

        sequence.token_ids = self.tokenizer.encode(prompt)

        while not self.stopping.should_stop(sequence):

            input_tensor = torch.tensor(
                [sequence.all_tokens],
                dtype=torch.long,
            )

            logits = self._forward(input_tensor)

            next_token = self._sample(
                logits,
                config,
            )

            sequence.append_token(next_token)

        return self._decode(sequence.all_tokens)


    def _forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        """

        return self.model.forward(
            input_ids=token_ids,
        )

    def _sample(
        self,
        logits: torch.Tensor,
        config: GenerationConfig,
    ) -> int:
        """
        Sample the next token using the configured strategy.
        """

        next_token_logits = logits[:, -1, :]

        return self.sampler.sample(
            next_token_logits,
            config,
        )

    def _decode(
        self,
        token_ids: list[int],
    ) -> str:
        """
        Decode generated token ids.
        """

        return self.tokenizer.decode(token_ids)