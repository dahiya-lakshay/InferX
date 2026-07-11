import torch

from app.engine.embedding import TokenEmbedding
from app.engine.positional_encoding import PositionalEncoding


def test_positional_encoding_shape():

    embedding = TokenEmbedding(
        vocab_size=100,
        embedding_dim=64,
    )

    position = PositionalEncoding(
        embedding_dim=64,
    )

    tokens = torch.tensor(
        [
            [1, 2, 3, 4]
        ]
    )

    x = embedding(tokens)

    output = position(x)

    assert output.shape == (
        1,
        4,
        64,
    )


def test_positional_encoding_changes_values():

    embedding = TokenEmbedding(
        vocab_size=100,
        embedding_dim=32,
    )

    position = PositionalEncoding(32)

    tokens = torch.tensor([[1, 2]])

    x = embedding(tokens)

    y = position(x)

    assert not torch.equal(x, y)