"""
Unit tests for InferX tokenizer.
"""

import tempfile

from app.tokenizer.tokenizer import Tokenizer
from app.tokenizer.utils import (
    normalize_text,
    preprocess_text,
    remove_punctuation,
)
from app.tokenizer.vocabulary import Vocabulary


def test_add_token():

    vocab = Vocabulary()

    vocab.add_token("hello")

    assert vocab.contains("hello")


def test_tokenize():

    tokenizer = Tokenizer()

    tokens = tokenizer.tokenize("Hello InferX")

    assert tokens == ["Hello", "InferX"]


def test_build_vocabulary():

    tokenizer = Tokenizer()

    tokenizer.build_vocabulary(
        [
            "hello world",
            "hello inferx",
        ]
    )

    assert tokenizer.vocab.contains("hello")

    assert tokenizer.vocab.contains("world")


def test_encode_decode():

    tokenizer = Tokenizer()

    tokenizer.build_vocabulary(
        [
            "hello inferx",
        ]
    )

    ids = tokenizer.encode("hello inferx")

    text = tokenizer.decode(ids)

    assert text == "hello inferx"


def test_normalize():

    assert (
        normalize_text(
            "  Hello    World "
        )
        == "hello world"
    )


def test_remove_punctuation():

    assert (
        remove_punctuation(
            "Hello, World!"
        )
        == "Hello World"
    )


def test_preprocess():

    assert (
        preprocess_text(
            " Hello, InferX!! "
        )
        == "hello inferx"
    )


def test_save_load_vocab():

    vocab = Vocabulary()

    vocab.add_token("inferx")

    with tempfile.NamedTemporaryFile() as file:

        vocab.save(file.name)

        loaded = Vocabulary.load(file.name)

        assert loaded.contains("inferx")