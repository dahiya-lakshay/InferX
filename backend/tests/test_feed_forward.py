import torch

from app.engine.feed_forward import FeedForward


def test_output_shape():

    ffn = FeedForward(
        embedding_dim=512,
        hidden_dim=2048,
    )

    x = torch.randn(
        2,
        16,
        512,
    )

    y = ffn(x)

    assert y.shape == x.shape


def test_gradient_flow():

    ffn = FeedForward(
        embedding_dim=256,
        hidden_dim=1024,
    )

    x = torch.randn(
        2,
        10,
        256,
        requires_grad=True,
    )

    y = ffn(x)

    loss = y.mean()

    loss.backward()

    assert x.grad is not None


def test_output_dtype():

    ffn = FeedForward(
        embedding_dim=128,
        hidden_dim=512,
    )

    x = torch.randn(
        1,
        5,
        128,
    )

    y = ffn(x)

    assert y.dtype == torch.float32