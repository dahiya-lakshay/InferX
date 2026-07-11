import torch

from app.engine.layer_norm import LayerNorm


def test_output_shape():

    norm = LayerNorm(512)

    x = torch.randn(2, 16, 512)

    y = norm(x)

    assert y.shape == x.shape


def test_gradient():

    norm = LayerNorm(256)

    x = torch.randn(
        2,
        10,
        256,
        requires_grad=True,
    )

    y = norm(x)

    loss = y.mean()

    loss.backward()

    assert x.grad is not None


def test_gamma_beta():

    norm = LayerNorm(128)

    assert norm.gamma.shape == (128,)
    assert norm.beta.shape == (128,)