import torch

from app.engine.batch import Batch
from app.engine.generation_config import GenerationConfig
from app.engine.request import GenerationRequest
from app.engine.sequence import Sequence


def make_sequence(tokens):

    request = GenerationRequest(
        request_id="1",
        prompt="",
        config=GenerationConfig(),
    )

    sequence = Sequence(
        request=request,
    )

    sequence.token_ids = tokens

    return sequence


def test_batch_size():

    batch = Batch(
        [
            make_sequence([1, 2]),
            make_sequence([3]),
        ]
    )

    assert batch.batch_size == 2


def test_max_sequence_length():

    batch = Batch(
        [
            make_sequence([1]),
            make_sequence([1, 2, 3]),
        ]
    )

    assert batch.max_sequence_length == 3


def test_build_input_ids():

    batch = Batch(
        [
            make_sequence([1, 2]),
            make_sequence([3]),
        ]
    )

    tensor = batch.build_input_ids()

    assert isinstance(
        tensor,
        torch.Tensor,
    )

    assert tensor.shape == (
        2,
        2,
    )