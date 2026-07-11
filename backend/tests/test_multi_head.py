import pytest
import torch

from app.engine.attention.multi_head import MultiHeadAttention


def test_output_shape():

    attention = MultiHeadAttention(
        embedding_dim=512,
        num_heads=8,
    )

    x = torch.randn(2, 16, 512)

    output, weights = attention(x)

    assert output.shape == (2, 16, 512)

    assert weights.shape == (2, 8, 16, 16)


def test_invalid_num_heads():

    with pytest.raises(ValueError):

        MultiHeadAttention(
            embedding_dim=510,
            num_heads=8,
        )


def test_gradient_flow():

    attention = MultiHeadAttention(
        embedding_dim=256,
        num_heads=8,
    )

    x = torch.randn(
        2,
        10,
        256,
        requires_grad=True,
    )

    output, _ = attention(x)

    loss = output.mean()

    loss.backward()

    assert x.grad is not None


def test_mask_support():

    attention = MultiHeadAttention(
        embedding_dim=128,
        num_heads=8,
    )

    x = torch.randn(
        2,
        6,
        128,
    )

    mask = torch.ones(
        2,
        1,
        6,
        6,
    )

    output, weights = attention(
        x,
        mask,
    )

    assert output.shape == (
        2,
        6,
        128,
    )

    assert weights.shape == (
        2,
        8,
        6,
        6,
    )