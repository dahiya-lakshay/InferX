"""
InferX - Sampling Primitives

This module implements the decoding algorithms used by modern
large language models during inference.

Algorithms:
- Stable Softmax
- Temperature Scaling
- Greedy Sampling
- Random Sampling
- Top-k Sampling
- Top-p (Nucleus) Sampling
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """
    Compute a numerically stable softmax.

    Parameters
    ----------
    logits : np.ndarray
        Raw logits from the model.

    Returns
    -------
    np.ndarray
        Probability distribution.
    """

    logits = np.asarray(logits, dtype=np.float64)

    shifted = logits - np.max(logits)
    exp = np.exp(shifted)

    return exp / np.sum(exp)


def apply_temperature(
    logits: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Apply temperature scaling.

    Lower temperature -> more deterministic.
    Higher temperature -> more random.
    """

    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    return logits / temperature


def greedy_sample(logits: np.ndarray) -> int:
    """
    Select the token with the maximum probability.
    """

    probs = stable_softmax(logits)

    return int(np.argmax(probs))


def random_sample(logits: np.ndarray) -> int:
    """
    Sample a token according to the probability distribution.
    """

    probs = stable_softmax(logits)

    return int(np.random.choice(len(probs), p=probs))


def top_k_sample(
    logits: np.ndarray,
    k: int,
) -> Tuple[int, np.ndarray]:
    """
    Sample only from the top-k highest probability tokens.
    """

    if k <= 0:
        raise ValueError("k must be positive")

    probs = stable_softmax(logits)

    k = min(k, len(probs))

    top_indices = np.argpartition(probs, -k)[-k:]

    top_probs = probs[top_indices]
    top_probs /= np.sum(top_probs)

    token = int(np.random.choice(top_indices, p=top_probs))

    return token, top_indices


def top_p_sample(
    logits: np.ndarray,
    p: float = 0.9,
) -> Tuple[int, np.ndarray]:
    """
    Nucleus (Top-p) sampling.

    Keeps the smallest set of tokens whose cumulative
    probability exceeds p.
    """

    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")

    probs = stable_softmax(logits)

    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    cumulative = np.cumsum(sorted_probs)

    cutoff = np.searchsorted(cumulative, p) + 1

    candidate_indices = sorted_indices[:cutoff]
    candidate_probs = sorted_probs[:cutoff]

    candidate_probs /= np.sum(candidate_probs)

    token = int(
        np.random.choice(
            candidate_indices,
            p=candidate_probs,
        )
    )

    return token, candidate_indices