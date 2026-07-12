import torch

from app.models.hf_adapter import HFAdapter


def test_forward():

    adapter = HFAdapter()

    input_ids = torch.randint(
        0,
        100,
        (2, 8),
    )

    logits = adapter.forward(
        input_ids=input_ids,
    )

    assert logits.ndim == 3

    assert logits.shape[0] == 2

    assert logits.shape[1] == 8