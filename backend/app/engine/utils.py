"""
InferX - Sampling Utilities

Reusable helper functions for validating logits,
normalizing probability distributions, masking tokens,
and reproducibility.

These utilities are shared across the inference engine.
"""

from __future__ import annotations

import random
from typing import Iterable

import numpy as np


def validate_logits(logits: np.ndarray) -> np.ndarray:
    """
    Validate logits before sampling.
    """

    logits = np.asarray(logits, dtype=np.float64)

    if logits.ndim != 1:
        raise ValueError("logits must be a 1-D array")

    if logits.size == 0:
        raise ValueError("logits cannot be empty")

    if np.any(np.isnan(logits)):
        raise ValueError("logits contain NaN values")

    if np.any(np.isinf(logits)):
        raise ValueError("logits contain infinite values")

    return logits


def normalize(probabilities: np.ndarray) -> np.ndarray:
    """
    Normalize a probability distribution.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)

    total = probabilities.sum()

    if total <= 0:
        raise ValueError("sum of probabilities must be positive")

    return probabilities / total


def mask_logits(
    logits: np.ndarray,
    indices_to_keep: Iterable[int],
) -> np.ndarray:
    """
    Mask all logits except the provided indices.
    """

    logits = validate_logits(logits)

    masked = np.full_like(logits, -np.inf)

    indices = np.asarray(list(indices_to_keep), dtype=int)

    masked[indices] = logits[indices]

    return masked


def set_seed(seed: int) -> None:
    """
    Set NumPy and Python random seeds.
    """

    np.random.seed(seed)
    random.seed(seed)


def entropy(probabilities: np.ndarray) -> float:
    """
    Compute Shannon entropy of a probability distribution.
    """

    probabilities = normalize(probabilities)

    epsilon = 1e-12

    return float(
        -np.sum(probabilities * np.log(probabilities + epsilon))
    )