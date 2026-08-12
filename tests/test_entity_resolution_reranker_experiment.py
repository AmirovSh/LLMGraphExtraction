from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from devtools.entity_resolution_reranker import (
    Benchmark, DirectionalPair, EntityProfile, PairCase, RerankerClient, RerankerConfig,
    RerankerResult, aggregate_scores, apply_guard, assert_group_isolation,
    directional_pairs, join_directional_results, strategy_decision, validate_score,
)


ROOT = Path(__file__).resolve().parents[1]


class Response:
    def __init__(self, score: float): self.score = score
    def raise_for_status(self): return None
    def json(self): return {"results": [{"index": 0, "relevance_score": self.score}]}


def case(case_id: str = "pair", *, right_group: str = "g") -> PairCase:
    return PairCase(
        case_id=case_id,
        entity_a=EntityProfile(canonical_name="Alpha", entity_type="ORG", aliases=["A"], graph_group="g"),
        entity_b=EntityProfile(canonical_name="Beta", entity_type="ORG", aliases=["B"], graph_group=right_group),
        expected="SPLIT", source="test", reason="distinct",
    )


def test_request_construction_preserves_pair_and_direction_order() -> None:
    captured = []
    scores = iter((0.8, 0.7))
    client = RerankerClient(
        RerankerConfig("https://rerank.invalid/v1", "secret", trust_env=False),
        post=lambda *args, **kwargs: captured.append((args, kwargs)) or Response(next(scores)),
    )
    pairs = directional_pairs([case()], "R2")
    results = client.score_many(pairs, batch_size=2)
    assert [(item.pair_id, item.direction) for item in results] == [("pair", "ab"), ("pair", "ba")]
    assert [item[0][0] for item in captured] == ["https://rerank.invalid/v1/rerank"] * 2
    assert [item[1]["json"]["query"] for item in captured] == [pairs[0].query, pairs[1].query]
    assert all(item[1]["json"]["model"] == "bge-reranker-v2-m3" for item in captured)
    assert all(item[1]["trust_env"] is False for item in captured)


def test_symmetric_aggregation_and_direction_validation() -> None:
    assert aggregate_scores(0.2, 0.8) == {"mean": 0.5, "min": 0.2, "max": 0.8}
    joined = join_directional_results([
        RerankerResult("p", "ab", 0.2, 0.1), RerankerResult("p", "ba", 0.8, 0.1),
    ])
    assert joined["p"]["mean"] == 0.5
    with pytest.raises(ValueError, match="missing direction"):
        join_directional_results([RerankerResult("p", "ab", 0.2, 0.1)])


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "0.5", True])
def test_invalid_scores_are_rejected(value) -> None:
    with pytest.raises(ValueError):
        validate_score(value)


def test_graph_group_isolation_rejects_cross_group_candidate() -> None:
    cross = case("cross", right_group="other")
    assert_group_isolation([cross], [])
    with pytest.raises(ValueError, match="cross-graph-group"):
        assert_group_isolation([cross], ["cross"])


def test_guard_can_only_demote_existing_merges_and_is_deterministic() -> None:
    decisions = [
        {"left_local_id": "M001", "right_local_id": "M002", "decision": "MERGE", "reason": "base"},
        {"left_local_id": "M001", "right_local_id": "M003", "decision": "UNRESOLVED", "reason": "base"},
    ]
    expected = [
        {"left_local_id": "M001", "right_local_id": "M002", "decision": "KEEP_SEPARATE", "reason": "experimental reranker precision guard"},
        decisions[1],
    ]
    assert apply_guard(decisions, set()) == expected
    assert apply_guard(decisions, set()) == expected
    assert len(apply_guard(decisions, set())) == len(decisions)
    assert strategy_decision("SPLIT", 1.0, strategy="R-GUARD", threshold=0.5) == "SPLIT"


def test_ambiguous_strategy_changes_only_unresolved_candidates() -> None:
    assert strategy_decision("UNRESOLVED", 0.9, strategy="R-AMBIGUOUS", threshold=0.8) == "MERGE"
    assert strategy_decision("UNRESOLVED", 0.7, strategy="R-AMBIGUOUS", threshold=0.8) == "SPLIT"
    assert strategy_decision("MERGE", 0.0, strategy="R-AMBIGUOUS", threshold=0.8) == "MERGE"
    assert strategy_decision("SPLIT", 1.0, strategy="R-AMBIGUOUS", threshold=0.8) == "SPLIT"
    assert strategy_decision("INELIGIBLE", 1.0, strategy="R-AMBIGUOUS", threshold=0.8) == "INELIGIBLE"


def test_benchmark_contract_and_production_configuration_are_unchanged() -> None:
    benchmark = Benchmark.model_validate_json(
        (ROOT / "devtools/fixtures/entity_resolution_reranker_pairs.json").read_text(encoding="utf-8")
    )
    assert {item.expected for item in benchmark.cases} == {"MERGE", "SPLIT"}
    assert any(item.entity_a.graph_group != item.entity_b.graph_group for item in benchmark.cases)
    config = (ROOT / "config/entity_resolution.yaml").read_text(encoding="utf-8")
    assert "model: bge-m3" in config
    assert "input_format: canonical_name_only" in config
    assert "candidate_similarity: 0.76" in config
    assert "automatic_merge_similarity: 0.90" in config
    assert "rerank" not in config.casefold()


def test_batch_result_order_is_stable_when_transport_finishes_out_of_order() -> None:
    pairs = [DirectionalPair(str(index), "q", "d", "ab") for index in range(4)]
    client = RerankerClient(
        RerankerConfig("https://rerank.invalid/v1", "secret"),
        post=lambda *_a, **_k: Response(0.5),
    )
    assert [item.pair_id for item in client.score_many(pairs, batch_size=4)] == ["0", "1", "2", "3"]
