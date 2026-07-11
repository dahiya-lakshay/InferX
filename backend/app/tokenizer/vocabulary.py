"""
InferX - Vocabulary

Manages the token <-> id mapping used by the tokenizer.
"""

from __future__ import annotations

import json
from pathlib import Path


class Vocabulary:
    """
    Simple vocabulary implementation.
    """

    SPECIAL_TOKENS = [
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<EOS>",
    ]

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

        for token in self.SPECIAL_TOKENS:
            self.add_token(token)

    def __len__(self) -> int:
        return len(self.token_to_id)

    def add_token(self, token: str) -> int:
        """
        Add a token to the vocabulary.
        """

        if token not in self.token_to_id:

            token_id = len(self.token_to_id)

            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

        return self.token_to_id[token]

    def token_id(self, token: str) -> int:
        """
        Return the ID for a token.
        Unknown tokens map to <UNK>.
        """

        return self.token_to_id.get(
            token,
            self.token_to_id["<UNK>"],
        )

    def id_token(self, token_id: int) -> str:
        """
        Return the token for an ID.
        """

        return self.id_to_token.get(
            token_id,
            "<UNK>",
        )

    def contains(self, token: str) -> bool:
        """
        Check if a token exists.
        """

        return token in self.token_to_id

    def save(self, path: str | Path) -> None:
        """
        Save vocabulary to JSON.
        """

        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                self.token_to_id,
                file,
                indent=4,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """
        Load vocabulary from JSON.
        """

        vocab = cls()

        vocab.token_to_id.clear()
        vocab.id_to_token.clear()

        with open(path, "r", encoding="utf-8") as file:
            mapping = json.load(file)

        for token, token_id in mapping.items():

            token_id = int(token_id)

            vocab.token_to_id[token] = token_id
            vocab.id_to_token[token_id] = token

        return vocab