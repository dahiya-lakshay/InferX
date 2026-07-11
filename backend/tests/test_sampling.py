"""
Unit tests for InferX sampling primitives.
"""

import numpy as np
import pytest

from app.engine.sampling import (
    apply_temperature,
    greedy_sample,
    random_sample,
    stable_softmax,
    top_k_sample,
    top_p_sample,
)


def test_softmax_sum():
    logits = np.array([1.0, 2.0, 3.0])

    probs = stable_softmax(logits)

    assert np.isclose(np.sum(probs), 1.0)


def test_softmax_positive():
    logits = np.array([5.0, 2.0, 0.0])

    probs = stable_softmax(logits)

    assert np.all(probs >= 0)


def test_temperature():
    logits = np.array([2.0, 4.0, 6.0])

    scaled = apply_temperature(logits, 2.0)

    assert np.allclose(
        scaled,
        np.array([1.0, 2.0, 3.0])
    )


def test_temperature_error():

    with pytest.raises(ValueError):
        apply_temperature(np.array([1, 2]), 0)


def test_greedy():

    logits = np.array([0.1, 0.2, 10.0])

    token = greedy_sample(logits)

    assert token == 2


def test_random():

    logits = np.array([2.0, 3.0, 4.0])

    token = random_sample(logits)

    assert token in [0, 1, 2]


def test_top_k():

    logits = np.array([1.0, 8.0, 7.0, 6.0])

    token, indices = top_k_sample(logits, 2)

    assert token in indices

    assert len(indices) == 2


def test_top_p():

    logits = np.array([8.0, 7.0, 2.0, 1.0])

    token, indices = top_p_sample(logits, 0.90)

    assert token in indices


def test_invalid_top_p():

    with pytest.raises(ValueError):
        top_p_sample(np.array([1, 2]), 1.5)


def test_invalid_top_k():

    with pytest.raises(ValueError):
        top_k_sample(np.array([1, 2]), 0)