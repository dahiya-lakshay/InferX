import pytest
import torch

from app.models.base_model import BaseModelAdapter


class DummyAdapter(BaseModelAdapter):

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        kv_cache=None,
        use_cache=False,
    ) -> torch.Tensor:

        batch, seq = input_ids.shape

        return torch.zeros(
            batch,
            seq,
            10,
        )


def test_forward():

    adapter = DummyAdapter()

    logits = adapter.forward(
        torch.tensor([[1, 2, 3]])
    )

    assert logits.shape == (
        1,
        3,
        10,
    )