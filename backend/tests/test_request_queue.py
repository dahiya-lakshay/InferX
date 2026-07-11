import pytest

from app.engine.generation_config import GenerationConfig
from app.engine.request import GenerationRequest
from app.engine.request_queue import RequestQueue


def test_enqueue():

    queue = RequestQueue()

    request = GenerationRequest(
        request_id="1",
        prompt="Hello",
        config=GenerationConfig(),
    )

    queue.enqueue(request)

    assert queue.size() == 1


def test_fifo():

    queue = RequestQueue()

    request1 = GenerationRequest(
        "1",
        "Hello",
        GenerationConfig(),
    )

    request2 = GenerationRequest(
        "2",
        "InferX",
        GenerationConfig(),
    )

    queue.enqueue(request1)

    queue.enqueue(request2)

    assert queue.dequeue().request_id == "1"

    assert queue.dequeue().request_id == "2"


def test_empty_queue():

    queue = RequestQueue()

    assert queue.is_empty()


def test_peek():

    queue = RequestQueue()

    request = GenerationRequest(
        "1",
        "Hello",
        GenerationConfig(),
    )

    queue.enqueue(request)

    assert queue.peek().request_id == "1"


def test_dequeue_empty():

    queue = RequestQueue()

    with pytest.raises(IndexError):

        queue.dequeue()