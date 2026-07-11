"""
InferX - Tokenizer Utilities

Utility functions for preprocessing and cleaning text
before tokenization.
"""

from __future__ import annotations

import re
import string
from pathlib import Path


def normalize_text(text: str) -> str:
    """
    Normalize text by converting to lowercase and
    removing extra whitespace.
    """

    text = text.lower()

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation characters.
    """

    return text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation,
        )
    )


def preprocess_text(text: str) -> str:
    """
    Complete preprocessing pipeline.
    """

    text = normalize_text(text)

    text = remove_punctuation(text)

    return text


def load_corpus(path: str | Path) -> list[str]:
    """
    Load a text corpus.

    Each line is treated as one document.
    """

    with open(path, "r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def save_corpus(
    texts: list[str],
    path: str | Path,
) -> None:
    """
    Save a corpus.
    """

    with open(path, "w", encoding="utf-8") as file:

        for text in texts:

            file.write(text + "\n")


def vocabulary_statistics(
    vocabulary,
) -> dict[str, int]:
    """
    Return basic vocabulary statistics.
    """

    return {
        "vocabulary_size": len(vocabulary),
        "special_tokens": len(vocabulary.SPECIAL_TOKENS),
    }