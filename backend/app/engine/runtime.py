"""
InferX - Runtime

Coordinates the scheduler and decode step.
"""

from __future__ import annotations

from app.engine.batch import Batch
from app.engine.decode_step import DecodeStep
from app.engine.request_queue import RequestQueue
from app.engine.scheduler import Scheduler


class Runtime:
    """
    Executes the inference runtime loop.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        decoder: DecodeStep,
        request_queue: RequestQueue,
    ) -> None:

        self.scheduler = scheduler
        self.decoder = decoder
        self.request_queue = request_queue

    def step(self) -> None:
        """
        Execute one scheduling iteration.
        """

        self.scheduler.admit(
            self.request_queue,
        )

        if not self.scheduler.has_active_sequences():
            return

        batch = Batch(
            self.scheduler.active_sequences,
        )

        self.decoder.forward(batch)

        self.scheduler.remove_finished()