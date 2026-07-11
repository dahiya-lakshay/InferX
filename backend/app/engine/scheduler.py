"""
InferX - Scheduler

Moves requests from the waiting queue into
active generation sequences.
"""

from __future__ import annotations

from app.engine.request_queue import RequestQueue
from app.engine.sequence import Sequence


class Scheduler:
    """
    Manages active generation sequences.
    """

    def __init__(
        self,
        max_active_sequences: int = 8,
    ) -> None:

        self.max_active_sequences = max_active_sequences

        self.active_sequences: list[Sequence] = []

    def admit(
        self,
        request_queue: RequestQueue,
    ) -> None:
        """
        Admit new requests until capacity is reached.
        """

        while (
            not request_queue.is_empty()
            and len(self.active_sequences)
            < self.max_active_sequences
        ):

            request = request_queue.dequeue()

            sequence = Sequence(
                request=request,
            )

            self.active_sequences.append(
                sequence,
            )

    def remove_finished(self) -> None:
        """
        Remove completed sequences.
        """

        self.active_sequences = [
            sequence
            for sequence in self.active_sequences
            if not sequence.is_finished
        ]

    def has_active_sequences(self) -> bool:

        return len(self.active_sequences) > 0