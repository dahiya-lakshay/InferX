from app.engine.generation_config import GenerationConfig
from app.engine.request import GenerationRequest
from app.engine.request_queue import RequestQueue
from app.engine.scheduler import Scheduler


def test_scheduler_admit():

    queue = RequestQueue()

    scheduler = Scheduler(
        max_active_sequences=2,
    )

    queue.enqueue(
        GenerationRequest(
            "1",
            "Hello",
            GenerationConfig(),
        )
    )

    scheduler.admit(queue)

    assert len(scheduler.active_sequences) == 1

    assert queue.is_empty()


def test_scheduler_capacity():

    queue = RequestQueue()

    scheduler = Scheduler(
        max_active_sequences=2,
    )

    for i in range(5):

        queue.enqueue(
            GenerationRequest(
                str(i),
                "InferX",
                GenerationConfig(),
            )
        )

    scheduler.admit(queue)

    assert len(scheduler.active_sequences) == 2

    assert queue.size() == 3


def test_remove_finished():

    queue = RequestQueue()

    scheduler = Scheduler()

    queue.enqueue(
        GenerationRequest(
            "1",
            "Hello",
            GenerationConfig(),
        )
    )

    scheduler.admit(queue)

    scheduler.active_sequences[0].is_finished = True

    scheduler.remove_finished()

    assert scheduler.has_active_sequences() is False