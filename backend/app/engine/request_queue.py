"""
InferX - Request Queue

FIFO queue for incoming generation requests.
"""

from __future__ import annotations

from collections import deque

from app.engine.request import GenerationRequest


class RequestQueue:
    """
    Queue of pending generation requests.
    """

    def __init__(self) -> None:

        self._queue = deque()

    def enqueue(
        self,
        request: GenerationRequest,
    ) -> None:

        self._queue.append(request)

    def dequeue(
        self,
    ) -> GenerationRequest:

        if self.is_empty():
            raise IndexError(
                "Request queue is empty."
            )

        return self._queue.popleft()

    def peek(
        self,
    ) -> GenerationRequest:

        if self.is_empty():
            raise IndexError(
                "Request queue is empty."
            )

        return self._queue[0]

    def is_empty(
        self,
    ) -> bool:

        return len(self._queue) == 0

    def size(
        self,
    ) -> int:

        return len(self._queue)