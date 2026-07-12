"""
Unit tests for InferX application settings.
"""

from app.core.settings import (
    HF_MODEL_NAME,
    MODEL_BACKEND,
)


def test_model_backend():

    assert MODEL_BACKEND in {
        "hf",
        "transformer",
    }


def test_hf_model_name():

    assert isinstance(
        HF_MODEL_NAME,
        str,
    )

    assert len(HF_MODEL_NAME) > 0