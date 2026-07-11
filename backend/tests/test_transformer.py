import torch

from app.engine.transformer import Transformer


def test_output_shape():

    model = Transformer(
        vocab_size=1000,
        embedding_dim=256,
        num_heads=8,
        hidden_dim=1024,
        num_layers=4,
    )

    tokens = torch.randint(
        0,
        1000,
        (2, 12),
    )

    logits = model(tokens)

    assert logits.shape == (
        2,
        12,
        1000,
    )


def test_gradient():

    model = Transformer(
        vocab_size=500,
        embedding_dim=128,
        num_heads=8,
        hidden_dim=512,
        num_layers=2,
    )

    tokens = torch.randint(
        0,
        500,
        (2, 8),
    )

    logits = model(tokens)

    loss = logits.mean()

    loss.backward()

    for parameter in model.parameters():

        assert parameter.grad is not None