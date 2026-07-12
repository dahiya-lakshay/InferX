"""
InferX - Bootstrap

Initializes application resources.
"""

from __future__ import annotations

from app.core.engine_registry import engine_registry
from app.engine.inference import InferenceEngine
from app.engine.transformer import Transformer
from app.models.transformer_adapter import TransformerAdapter
from app.tokenizer.tokenizer import Tokenizer


def bootstrap() -> None:
    """
    Initialize application resources.
    """

    tokenizer = Tokenizer()

    model = Transformer(
        vocab_size=5000,
        embedding_dim=256,
        num_heads=8,
        num_layers=4,
        hidden_dim=1024,
    )

    adapter = TransformerAdapter(model)

    engine = InferenceEngine(
        model=adapter,
        tokenizer=tokenizer,
    )

    engine_registry.register(engine)