"""
Unit tests for InferX sampling primitives.
"""

import pytest
import torch

from app.engine.sampling import (
    apply_temperature,
    greedy_sample,
    random_sample,
    stable_softmax,
    top_k_sample,
)


def test_softmax_sum():

    logits = torch.tensor([1.0, 2.0, 3.0])

    probs = stable_softmax(logits)

    assert torch.isclose(
        probs.sum(),
        torch.tensor(1.0),
    )


def test_softmax_positive():

    logits = torch.tensor([5.0, 2.0, 0.0])

    probs = stable_softmax(logits)

    assert torch.all(probs >= 0)


def test_temperature():

    logits = torch.tensor([2.0, 4.0, 6.0])

    scaled = apply_temperature(
        logits,
        2.0,
    )

    expected = torch.tensor(
        [1.0, 2.0, 3.0]
    )

    assert torch.allclose(
        scaled,
        expected,
    )


def test_temperature_error():

    with pytest.raises(ValueError):

        apply_temperature(
            torch.tensor([1.0, 2.0]),
            0,
        )


def test_greedy():

    logits = torch.tensor(
        [0.1, 0.2, 10.0]
    )

    token = greedy_sample(logits)

    assert token.item() == 2


def test_random():

    logits = torch.tensor(
        [2.0, 3.0, 4.0]
    )

    token = random_sample(logits)

    assert token.item() in [0, 1, 2]


def test_top_k():

    logits = torch.tensor(
        [1.0, 8.0, 7.0, 6.0]
    )

    token = top_k_sample(
        logits,
        2,
    )

    assert token.item() in [1, 2]


def test_invalid_top_k():

    with pytest.raises(ValueError):

        top_k_sample(
            torch.tensor([1.0, 2.0]),
            0,
        )