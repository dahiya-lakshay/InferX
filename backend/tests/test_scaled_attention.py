import torch

from app.engine.attention.scaled_attention import (
    ScaledDotProductAttention,
)


def test_output_shape():

    attention = ScaledDotProductAttention()

    query = torch.randn(2, 8, 10, 64)
    key = torch.randn(2, 8, 10, 64)
    value = torch.randn(2, 8, 10, 64)

    output, weights = attention(
        query,
        key,
        value,
    )

    assert output.shape == (2, 8, 10, 64)
    assert weights.shape == (2, 8, 10, 10)


def test_attention_weights_sum_to_one():

    attention = ScaledDotProductAttention()

    query = torch.randn(1, 8, 6, 64)
    key = torch.randn(1, 8, 6, 64)
    value = torch.randn(1, 8, 6, 64)

    _, weights = attention(
        query,
        key,
        value,
    )

    sums = weights.sum(dim=-1)

    assert torch.allclose(
        sums,
        torch.ones_like(sums),
        atol=1e-6,
    )


def test_masking():

    attention = ScaledDotProductAttention()

    query = torch.randn(1, 4, 5, 32)
    key = torch.randn(1, 4, 5, 32)
    value = torch.randn(1, 4, 5, 32)

    mask = torch.ones(1, 1, 5, 5)
    mask[:, :, :, -1] = 0

    _, weights = attention(
        query,
        key,
        value,
        mask,
    )

    assert torch.all(weights[..., -1] < 1e-6)


def test_gradient_flow():

    attention = ScaledDotProductAttention()

    query = torch.randn(
        2,
        8,
        5,
        32,
        requires_grad=True,
    )

    key = torch.randn(
        2,
        8,
        5,
        32,
        requires_grad=True,
    )

    value = torch.randn(
        2,
        8,
        5,
        32,
        requires_grad=True,
    )

    output, _ = attention(
        query,
        key,
        value,
    )

    loss = output.mean()

    loss.backward()

    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None


def test_invalid_dimensions():

    attention = ScaledDotProductAttention()

    query = torch.randn(2, 10, 64)
    key = torch.randn(2, 10, 64)
    value = torch.randn(2, 10, 64)

    try:
        attention(query, key, value)
        assert False
    except ValueError:
        assert True