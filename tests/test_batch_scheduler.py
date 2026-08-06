from runtime.batch_scheduler import ExtractionQueueItem, FairRoundRobinScheduler


def item(document, graph, order):
    return ExtractionQueueItem("b", graph, document, f"S{order:03d}", order, 0, 1,
                               1, "p", "s", "profile")


def test_round_robin_prevents_large_document_starvation():
    items = [item("large", "g1", index) for index in range(1, 8)] + [
        item("small", "g2", 1)]
    scheduler = FairRoundRobinScheduler(items, global_limit=1,
        per_document_limit=1, per_graph_limit=1)
    observed = []
    while scheduler.pending:
        selected = scheduler.acquire()
        assert selected is not None
        observed.append(selected.document_id)
        scheduler.complete(selected)
    assert observed[:2] == ["large", "small"]


def test_all_concurrency_limits_are_enforced():
    items = [item("d1", "g1", 1), item("d1", "g1", 2),
             item("d2", "g1", 1), item("d3", "g2", 1)]
    scheduler = FairRoundRobinScheduler(items, global_limit=3,
        per_document_limit=1, per_graph_limit=1)
    first = scheduler.acquire()
    second = scheduler.acquire()
    third = scheduler.acquire()
    assert first.document_id == "d1"
    assert second.graph_id == "g2"
    assert third is None
    assert scheduler.peak_in_flight == 2
    scheduler.complete(first)
    assert scheduler.acquire().document_id == "d2"


def test_discard_graph_preserves_other_groups():
    scheduler = FairRoundRobinScheduler([item("d1", "bad", 1), item("d2", "good", 1)],
        global_limit=2, per_document_limit=1, per_graph_limit=1)
    assert scheduler.discard_graph("bad") == 1
    assert scheduler.acquire().graph_id == "good"
