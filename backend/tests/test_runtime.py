import torch
import torch.nn as nn

from app.engine.decode_step import DecodeStep
from app.engine.generation_config import GenerationConfig
from app.engine.request import GenerationRequest
from app.engine.request_queue import RequestQueue
from app.engine.runtime import Runtime
from app.engine.scheduler import Scheduler


class DummyModel(nn.Module):

    def forward(self, input_ids):

        batch, seq = input_ids.shape

        return torch.randn(
            batch,
            seq,
            100,
        )


def test_runtime_step():

    queue = RequestQueue()

    scheduler = Scheduler()

    decoder = DecodeStep(
        DummyModel(),
    )

    runtime = Runtime(
        scheduler,
        decoder,
        queue,
    )

    queue.enqueue(
        GenerationRequest(
            request_id="1",
            prompt="Hello",
            config=GenerationConfig(),
        )
    )

    runtime.step()

    assert len(
        scheduler.active_sequences
    ) == 1

    assert (
        scheduler.active_sequences[0]
        .current_length
        == 1
    )