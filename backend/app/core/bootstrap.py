"""
InferX - Bootstrap

Initializes application resources.
"""

from __future__ import annotations

from app.core.engine_registry import engine_registry
from app.core.settings import (
    HF_MODEL_NAME,
    MODEL_BACKEND,
)
from app.engine.inference import InferenceEngine
from app.engine.transformer import Transformer
from app.models.hf_adapter import HFAdapter
from app.models.transformer_adapter import TransformerAdapter
from app.tokenizer.tokenizer import Tokenizer


def bootstrap() -> None:
    """
    Initialize application resources.
    """

    if MODEL_BACKEND == "transformer":

        tokenizer = Tokenizer()

        model = Transformer(
            vocab_size=5000,
            embedding_dim=256,
            num_heads=8,
            num_layers=4,
            hidden_dim=1024,
        )

        adapter = TransformerAdapter(model)

    elif MODEL_BACKEND == "hf":

        adapter = HFAdapter(
            model_name=HF_MODEL_NAME,
        )

        tokenizer = adapter.tokenizer

    else:
        raise ValueError(
            f"Unknown model backend: {MODEL_BACKEND}"
        )

    engine = InferenceEngine(
        model=adapter,
        tokenizer=tokenizer,
    )

    engine_registry.register(engine)