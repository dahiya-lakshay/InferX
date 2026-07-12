import torch

from app.engine.transformer import Transformer
from app.models.transformer_adapter import (
    TransformerAdapter,
)


def test_forward():

    model = Transformer(
        vocab_size=100,
        embedding_dim=32,
        num_heads=4,
        hidden_dim=64,
        num_layers=2,
    )

    adapter = TransformerAdapter(model)

    logits = adapter.forward(
        torch.randint(
            0,
            100,
            (2, 8),
        )
    )

    assert logits.shape == (
        2,
        8,
        100,
    )