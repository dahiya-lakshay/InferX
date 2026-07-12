"""
Unit tests for InferX Sampler.
"""

import torch

from app.engine.generation_config import GenerationConfig
from app.engine.sampler import Sampler


def test_greedy_sampler():

    sampler = Sampler()

    logits = torch.tensor(
        [0.1, 0.5, 2.0]
    )

    token = sampler.sample(
        logits,
        GenerationConfig(
            do_sample=False,
        ),
    )

    assert token == 2


def test_top_k_sampler():

    sampler = Sampler()

    logits = torch.tensor(
        [1.0, 5.0, 4.0]
    )

    token = sampler.sample(
        logits,
        GenerationConfig(
            do_sample=True,
            top_k=2,
        ),
    )

    assert token in [1, 2]