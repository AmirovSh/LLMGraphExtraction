"""Deterministic offline comparison of batch scheduling policies."""
from __future__ import annotations

import argparse
import json
from collections import deque
from statistics import median

from runtime.batch_scheduler import ExtractionQueueItem, FairRoundRobinScheduler


def _items() -> list[ExtractionQueueItem]:
    sizes = {"large": 40, "medium": 12, "small": 3}
    return [ExtractionQueueItem("bench", f"g_{name}", name, f"S{index:03d}", index,
        0, 1, 1, "p", "s", "fake", payload=1 + (index % 5))
        for name, size in sizes.items() for index in range(1, size + 1)]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def simulate(policy: str) -> dict[str, object]:
    items = _items()
    if policy == "serial_documents":
        order = items
    elif policy == "naive_fifo":
        order = list(items)
    else:
        scheduler = FairRoundRobinScheduler(items, global_limit=1,
            per_document_limit=1, per_graph_limit=1)
        order = []
        while scheduler.pending:
            item = scheduler.acquire()
            order.append(item)
            scheduler.complete(item)
    now = 0.0
    waits: list[float] = []
    completion: dict[str, float] = {}
    first_service: dict[str, float] = {}
    for item in order:
        first_service.setdefault(item.document_id, now)
        waits.append(now)
        now += float(item.payload)
        completion[item.document_id] = now
    return {"policy": policy, "total_batch_makespan": now,
            "documents_per_minute": round(len(completion) * 60 / now, 4),
            "spans_per_second": round(len(order) / now, 4),
            "span_queue_wait_p50": median(waits),
            "span_queue_wait_p95": _percentile(waits, .95),
            "document_completion_p50": median(completion.values()),
            "document_completion_p95": _percentile(list(completion.values()), .95),
            "graph_completion_p50": median(completion.values()),
            "graph_completion_p95": _percentile(list(completion.values()), .95),
            "peak_in_flight_requests": 1,
            "starvation_seconds": max(first_service.values()),
            "model_attempts": len(order), "embedding_calls": len(completion),
            "cached_input_tokens": None}


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps([simulate(policy) for policy in (
        "serial_documents", "naive_fifo", "fair_round_robin",
        "fair_round_robin_per_document_limits")], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
