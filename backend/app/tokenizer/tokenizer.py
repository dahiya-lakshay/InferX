"""
InferX - Tokenizer

A simple whitespace tokenizer built on top of the Vocabulary class.
"""

from __future__ import annotations

from app.tokenizer.vocabulary import Vocabulary


class Tokenizer:
    """
    Simple tokenizer implementation.
    """

    def __init__(self, vocabulary: Vocabulary | None = None) -> None:

        self.vocab = vocabulary or Vocabulary()

    def tokenize(self, text: str) -> list[str]:
        """
        Split text into tokens.
        """

        text = text.strip()

        if not text:
            return []

        return text.split()

    def detokenize(self, tokens: list[str]) -> str:
        """
        Join tokens back into text.
        """

        return " ".join(tokens)

    def build_vocabulary(self, texts: list[str]) -> None:
        """
        Build vocabulary from a corpus.
        """

        for text in texts:

            tokens = self.tokenize(text)

            for token in tokens:
                self.vocab.add_token(token)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """
        Convert text into token IDs.
        """

        tokens = self.tokenize(text)

        ids = []

        if add_special_tokens:
            ids.append(self.vocab.token_id("<BOS>"))

        for token in tokens:
            ids.append(self.vocab.token_id(token))

        if add_special_tokens:
            ids.append(self.vocab.token_id("<EOS>"))

        return ids

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Convert token IDs back into text.
        """

        tokens = []

        special = {
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>",
        }

        for token_id in token_ids:

            token = self.vocab.id_token(token_id)

            if skip_special_tokens and token in special:
                continue

            tokens.append(token)

        return self.detokenize(tokens)

    @property
    def vocabulary_size(self) -> int:
        """
        Return vocabulary size.
        """

        return len(self.vocab)