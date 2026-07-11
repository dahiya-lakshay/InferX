"""
InferX - Generation Sequence

Represents the mutable state of an active generation request.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.engine.kv_cache import KVCache
from app.engine.request import GenerationRequest


@dataclass
class Sequence:
    """
    Active sequence being generated.
    """

    request: GenerationRequest

    token_ids: list[int] = field(default_factory=list)

    generated_tokens: list[int] = field(default_factory=list)

    kv_cache: KVCache = field(default_factory=KVCache)

    is_finished: bool = False

    def append_token(
        self,
        token_id: int,
    ) -> None:
        """
        Append a generated token.
        """

        self.generated_tokens.append(token_id)

    @property
    def current_length(self) -> int:
        """
        Current generated sequence length.
        """

        return len(self.generated_tokens)

    @property
    def all_tokens(self) -> list[int]:
        """
        Prompt + generated tokens.
        """

        return self.token_ids + self.generated_tokens