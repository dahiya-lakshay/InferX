"""
InferX - Stopping Criteria

Utilities for deciding when autoregressive
generation should terminate.
"""

from __future__ import annotations

from app.engine.sequence import Sequence


class StoppingCriteria:
    """
    Determines whether generation should stop.
    """

    def should_stop(
        self,
        sequence: Sequence,
    ) -> bool:

        config = sequence.request.config

        # Maximum generated tokens
        if (
            sequence.current_length
            >= config.max_new_tokens
        ):
            return True

        # EOS token
        if (
            config.eos_token_id is not None
            and sequence.generated_tokens
            and sequence.generated_tokens[-1]
            == config.eos_token_id
        ):
            return True

        return False