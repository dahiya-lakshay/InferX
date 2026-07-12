import torch
import torch.nn as nn

from app.engine.batch import Batch
from app.engine.decode_step import DecodeStep
from app.engine.generation_config import GenerationConfig
from app.engine.request import GenerationRequest
from app.engine.sequence import Sequence


class DummyModel(nn.Module):

    def forward(
        self,
        input_ids,
        attention_mask=None,
        kv_cache=None,
        use_cache=False,
    ):

        batch_size, sequence_length = input_ids.shape

        vocab_size = 50

        return torch.randn(
            batch_size,
            sequence_length,
            vocab_size,
        )


def make_sequence():

    request = GenerationRequest(
        request_id="1",
        prompt="",
        config=GenerationConfig(),
    )

    return Sequence(
        request=request,
        token_ids=[1, 2, 3],
    )


def test_decode_step():

    batch = Batch(
        [
            make_sequence(),
            make_sequence(),
        ]
    )

    decoder = DecodeStep(
        DummyModel(),
    )

    decoder.forward(batch)

    assert (
        batch.sequences[0].current_length
        == 1
    )

    assert (
        batch.sequences[1].current_length
        == 1
    )