"""Fair graph-aware scheduling for independent extraction requests."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class ExtractionQueueItem:
    batch_id: str
    graph_id: str
    document_id: str
    span_id: str
    document_order: int
    source_start: int
    source_end: int
    estimated_input_tokens: int
    prompt_hash: str
    schema_hash: str
    provider_profile: str
    priority: int = 0
    enqueue_timestamp: float = 0.0
    payload: Any = field(default=None, compare=False)


class FairRoundRobinScheduler:
    """Admission scheduler; callers must notify completion explicitly."""

    def __init__(self, items: Iterable[ExtractionQueueItem], *, global_limit: int,
                 per_document_limit: int, per_graph_limit: int) -> None:
        self.global_limit = global_limit
        self.per_document_limit = per_document_limit
        self.per_graph_limit = per_graph_limit
        self._queues: dict[str, deque[ExtractionQueueItem]] = {}
        for item in items:
            self._queues.setdefault(item.document_id, deque()).append(item)
        self._active_documents = deque(self._queues)
        self._document_in_flight = {key: 0 for key in self._queues}
        self._graph_in_flight: dict[str, int] = {}
        self.in_flight = 0
        self.peak_in_flight = 0

    @property
    def pending(self) -> bool:
        return any(self._queues.values())

    def acquire(self) -> ExtractionQueueItem | None:
        if self.in_flight >= self.global_limit or not self._active_documents:
            return None
        for index, document_id in enumerate(tuple(self._active_documents)):
            queue = self._queues[document_id]
            if not queue:
                continue
            item = queue[0]
            allowed = (
                self._document_in_flight[document_id] < self.per_document_limit
                and self._graph_in_flight.get(item.graph_id, 0) < self.per_graph_limit
            )
            if not allowed:
                continue
            del self._active_documents[index]
            self._active_documents.append(document_id)
            queue.popleft()
            self._document_in_flight[document_id] += 1
            self._graph_in_flight[item.graph_id] = self._graph_in_flight.get(item.graph_id, 0) + 1
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            return item
        return None

    def complete(self, item: ExtractionQueueItem) -> None:
        self._document_in_flight[item.document_id] -= 1
        self._graph_in_flight[item.graph_id] -= 1
        self.in_flight -= 1

    def discard_graph(self, graph_id: str) -> int:
        """Discard pending work for a failed group; in-flight work is unaffected."""
        discarded = 0
        for document_id, queue in self._queues.items():
            kept = deque(item for item in queue if item.graph_id != graph_id)
            discarded += len(queue) - len(kept)
            self._queues[document_id] = kept
        return discarded
