import torch

from app.engine.embedding import TokenEmbedding


def test_embedding_shape():

    vocab_size = 100

    embedding_dim = 64

    embedding = TokenEmbedding(
        vocab_size,
        embedding_dim,
    )

    tokens = torch.tensor(
        [
            [1, 5, 8, 9]
        ]
    )

    output = embedding(tokens)

    assert output.shape == (
        1,
        4,
        64,
    )


def test_embedding_dtype():

    embedding = TokenEmbedding(
        50,
        32,
    )

    tokens = torch.tensor([[1, 2]])

    output = embedding(tokens)

    assert output.dtype == torch.float32