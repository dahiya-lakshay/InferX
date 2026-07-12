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

import torch


def stable_softmax(
    logits: torch.Tensor,
) -> torch.Tensor:

    """
    Compute a numerically stable softmax.
    """

    shifted_logits = logits - logits.max(
        dim=-1,
        keepdim=True,
    ).values

    return torch.softmax(
        shifted_logits,
        dim=-1,
    )


def apply_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Apply temperature scaling.

    Lower temperature -> more deterministic.
    Higher temperature -> more random.
    """

    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    return logits / temperature


def greedy_sample(
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    Select the highest-probability token.

    Supports batched logits.
    """

    return torch.argmax(
        logits,
        dim=-1,
    )


def random_sample(
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    Sample according to the probability distribution.
    """

    probabilities = stable_softmax(
        logits,
    )

    return torch.multinomial(
        probabilities,
        num_samples=1,
    ).squeeze(-1)


def top_k_sample(
    logits: torch.Tensor,
    k: int,
) -> torch.Tensor:

    if k <= 0:
        raise ValueError(
            "k must be positive"
        )

    values, indices = torch.topk(
        logits,
        k,
        dim=-1,
    )

    probabilities = torch.softmax(
        values,
        dim=-1,
    )

    sampled = torch.multinomial(
        probabilities,
        num_samples=1,
    )

    return indices.gather(
        -1,
        sampled,
    ).squeeze(-1)


def top_p_sample(
    logits: torch.Tensor,
    p: float = 0.9,
) -> torch.Tensor:
    """
    Sample from the smallest set of tokens whose
    cumulative probability exceeds p.
    """

    if not 0 < p <= 1:
        raise ValueError(
            "p must be in (0, 1]"
        )

    probabilities = stable_softmax(
        logits,
    )

    sorted_probs, sorted_indices = torch.sort(
        probabilities,
        descending=True,
        dim=-1,
    )

    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=-1,
    )

    keep = cumulative_probs <= p

    keep[..., 0] = True

    filtered_probs = sorted_probs * keep

    filtered_probs = (
        filtered_probs
        / filtered_probs.sum(
            dim=-1,
            keepdim=True,
        )
    )

    sampled = torch.multinomial(
        filtered_probs,
        num_samples=1,
    )

    return sorted_indices.gather(
        -1,
        sampled,
    ).squeeze(-1)