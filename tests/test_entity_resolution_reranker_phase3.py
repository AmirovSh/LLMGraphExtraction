from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.build_entity_resolution_reranker_phase3_benchmark import build
from devtools.entity_resolution_reranker_phase3 import (
    FROZEN_PHASE2, assert_frozen_phase2, assert_split_isolation, candidate_sweep,
    confidence_intervals, fixed_phase3_split, freeze_expanded_config, freeze_manifest,
    guard_decision, symmetric_top_k, threshold_robustness, top_k_sweep,
)

ROOT = Path(__file__).resolve().parents[1]


def benchmark():
    definitions = json.loads((ROOT / "devtools/fixtures/entity_resolution_reranker_phase3_definitions.json").read_text())
    return build(definitions)


def test_v3_is_fresh_balanced_and_contains_required_hard_cases():
    cases = benchmark().cases
    primary = [c for c in cases if not c.diagnostic_only]
    assert len(primary) >= 180
    assert 0.34 <= sum(c.expected == "MERGE" for c in primary) / len(primary) <= 0.40
    assert sum(c.category == "abbreviation_expansion" for c in primary) >= 25
    assert sum(c.category == "abbreviation_hard_negative" for c in primary) >= 20
    assert sum(c.category == "same_surface_negative" for c in primary) >= 25
    assert all(c.deterministic_block == "EXPLICIT_DISTINCT" for c in primary if c.category == "same_surface_negative")
    assert all("\\" not in c.source_reference and ":\\" not in c.source_reference for c in cases)


def test_split_is_cluster_isolated_reproducible_and_persistable_before_scoring():
    cases = benchmark().cases
    split = fixed_phase3_split(cases)
    assert split == fixed_phase3_split(cases)
    assert_split_isolation(cases, split)
    manifest = freeze_manifest(benchmark(), split)
    assert manifest["partition_counts"]["REPLICATION_SET"] >= 60
    assert manifest["partition_counts"]["COUNTERFACTUAL_STRESS"] == 20
    assert len(manifest["benchmark_hash"]) == len(manifest["split_hash"]) == 64


def test_phase2_frozen_configuration_cannot_be_modified():
    assert_frozen_phase2(FROZEN_PHASE2)
    for key in FROZEN_PHASE2:
        changed = dict(FROZEN_PHASE2)
        changed[key] = "changed"
        with pytest.raises(ValueError, match="frozen"):
            assert_frozen_phase2(changed)


def test_replication_never_enters_expanded_calibration():
    split = fixed_phase3_split(benchmark().cases)
    replication = {case_id for case_id, part in split.items() if part == "REPLICATION_SET"}
    calibration = {case_id for case_id, part in split.items() if part.startswith("EXPANDED_")}
    assert replication.isdisjoint(calibration)


@pytest.mark.parametrize("block", ["VERSION_CONFLICT", "OPPOSITION_CONFLICT", "EXPLICIT_DISTINCT"])
def test_safety_blocks_outrank_saturated_reranker(block):
    row = {"same_graph_group": True, "deterministic_block": block, "baseline_decision": "SPLIT",
           "bge_similarity": .75, "ab": 1.0, "ba": 1.0}
    assert guard_decision(row, candidate_floor=.55, threshold=.9, aggregation="max", strategy="R-EXPANDED") == "SPLIT"


def test_same_surface_cross_group_is_never_eligible():
    row = {"same_graph_group": False, "baseline_decision": "SPLIT", "bge_similarity": 1.0, "ab": 1.0, "ba": 1.0}
    assert guard_decision(row, candidate_floor=.55, threshold=.9, aggregation="max", strategy="R-EXPANDED") == "INELIGIBLE"


def test_consensus_requires_both_directions_and_ambiguous_does_not_expand():
    row = {"same_graph_group": True, "baseline_decision": "SPLIT", "bge_similarity": .70, "ab": 1.0, "ba": .1}
    assert guard_decision(row, candidate_floor=.60, threshold=.9, aggregation="min", strategy="R-CONSENSUS") == "SPLIT"
    assert guard_decision(row, candidate_floor=.60, threshold=.9, aggregation="max", strategy="R-EXPANDED") == "MERGE"
    assert guard_decision(row, candidate_floor=.60, threshold=.9, aggregation="max", strategy="R-AMBIGUOUS") == "SPLIT"


def test_candidate_zones_topk_and_large_graph_bounding():
    rows = [{"case_id": str(i), "expected": "MERGE" if i % 7 == 0 else "SPLIT",
             "bge_similarity": .55 + i / 1000, "entity_a_key": f"e{i % 10}", "entity_b_key": f"e{10 + (i // 10)}"}
            for i in range(100)]
    assert [r["candidate_floor"] for r in candidate_sweep(rows, [.76, .72, .68, .64, .60, .55])] == [.76, .72, .68, .64, .60, .55]
    assert len(symmetric_top_k(rows, 5, .55)) < len(rows)
    results = top_k_sweep(rows, [5, 10, 20, 40], .55)
    assert [r["candidate_volume"] for r in results] == sorted(r["candidate_volume"] for r in results)


def test_expansion_factor_uses_same_safety_eligible_denominator():
    rows = [{"case_id": "safe", "expected": "MERGE", "bge_similarity": .80, "entity_a_key": "a", "entity_b_key": "b"},
            {"case_id": "blocked", "expected": "SPLIT", "bge_similarity": 1.0, "entity_a_key": "c", "entity_b_key": "d", "deterministic_block": "EXPLICIT_DISTINCT"},
            {"case_id": "expanded", "expected": "MERGE", "bge_similarity": .60, "entity_a_key": "e", "entity_b_key": "f"}]
    sweep = candidate_sweep(rows, [.76, .55])
    assert sweep[0]["expansion_factor"] == 1.0
    assert sweep[1]["expansion_factor"] == 2.0


def test_confidence_and_threshold_robustness_are_deterministic():
    rows = [{"expected": "MERGE", "decision": "MERGE", "max": .99997},
            {"expected": "SPLIT", "decision": "SPLIT", "max": .3}]
    assert confidence_intervals(rows, resamples=2000) == confidence_intervals(rows, resamples=2000)
    robust = threshold_robustness(rows, .9999665, "max")
    assert set(robust["near_threshold"]) == {"0.001", "0.0001", "1e-05"}


def test_held_out_config_is_complete_and_hashed():
    config = {"model": "bge-reranker-v2-m3", "input_variant": "R2", "aggregation": "min",
              "threshold": .99, "candidate_floor": .60, "top_k": 10,
              "deterministic_exclusions": ["version", "opposition"], "strategy_version": "R-CONSENSUS-v1"}
    frozen = freeze_expanded_config(config, benchmark_hash="a", split_hash="b")
    assert len(frozen["config_hash"]) == 64
    with pytest.raises(ValueError, match="missing"):
        freeze_expanded_config({}, benchmark_hash="a", split_hash="b")


def test_counterfactuals_are_diagnostic_only():
    cases = benchmark().cases
    split = fixed_phase3_split(cases)
    assert all(split[c.case_id] == "COUNTERFACTUAL_STRESS" for c in cases if c.diagnostic_only)
    assert all(not c.diagnostic_only for c in cases if split[c.case_id] in {"EXPANDED_CALIBRATION", "EXPANDED_VALIDATION", "EXPANDED_HELD_OUT"})


def test_production_runners_do_not_import_or_call_reranker():
    for path in (ROOT / "runtime/production_runner.py", ROOT / "runtime/batch_runner.py"):
        text = path.read_text(encoding="utf-8").casefold()
        assert "reranker" not in text and "entity_resolution_reranker" not in text
    config = (ROOT / "config/entity_resolution.yaml").read_text(encoding="utf-8")
    assert "candidate_similarity: 0.76" in config and "automatic_merge_similarity: 0.90" in config
    assert "input_format: canonical_name_only" in config and "top_k: 10" in config
