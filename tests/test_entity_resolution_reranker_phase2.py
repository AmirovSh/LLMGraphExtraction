from __future__ import annotations

from devtools.build_entity_resolution_reranker_phase2_benchmark import build
from devtools.entity_resolution_reranker_phase2 import (
    assert_no_cluster_leakage, bootstrap_intervals, calibrate_without_held_out,
    candidate_sweep, expanded_decision, fixed_cluster_split, top_k_sweep,
)
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

def benchmark():
    return build(json.loads((ROOT / "devtools/fixtures/entity_resolution_reranker_phase2_definitions.json").read_text()))

def test_fixed_split_has_no_cluster_leakage_and_is_immutable():
    cases=benchmark().cases; one=fixed_cluster_split(cases); two=fixed_cluster_split(cases)
    assert one==two; assert_no_cluster_leakage(cases,one)
    assert set(one.values())=={"CALIBRATION","VALIDATION","HELD_OUT_TEST","ISOLATION"}

def test_candidate_and_topk_sweeps_are_bounded():
    rows=[{"case_id":"a","expected":"MERGE","bge_similarity":.7,"entity_a_key":"a","entity_b_key":"b"},
          {"case_id":"b","expected":"SPLIT","bge_similarity":.6,"entity_a_key":"a","entity_b_key":"c"}]
    sweep=candidate_sweep(rows,[.76,.60],production_floor=.76)
    assert sweep[0]["candidate_pairs"]==0 and sweep[1]["candidate_pairs"]==2
    assert top_k_sweep(rows,[1],floor=.6)[0]["candidate_pairs"]>=1

def test_expanded_never_bypasses_isolation_or_deterministic_block():
    base={"baseline_decision":"SPLIT","bge_similarity":.7,"max":1.0,"same_graph_group":True}
    assert expanded_decision(base|{"isolation_only":True},production_floor=.76,candidate_floor=.6,threshold=.9,aggregation="max")=="INELIGIBLE"
    assert expanded_decision(base|{"deterministic_block":True},production_floor=.76,candidate_floor=.6,threshold=.9,aggregation="max")=="SPLIT"
    assert expanded_decision(base,production_floor=.76,candidate_floor=.6,threshold=.9,aggregation="max")=="MERGE"

def test_calibration_rejects_held_out_labels():
    with pytest.raises(ValueError,match="held-out"):
        calibrate_without_held_out([{"partition":"HELD_OUT_TEST"}],production_floor=.76,floors=[.6])

def test_bootstrap_is_deterministic():
    rows=[{"decision":"MERGE","expected":"MERGE"},{"decision":"SPLIT","expected":"SPLIT"}]
    assert bootstrap_intervals(rows,"decision",resamples=100)==bootstrap_intervals(rows,"decision",resamples=100)

def test_production_settings_and_relations_are_untouched():
    text=(ROOT/"config/entity_resolution.yaml").read_text()
    assert "candidate_similarity: 0.76" in text and "automatic_merge_similarity: 0.90" in text
    assert "input_format: canonical_name_only" in text and "rerank" not in text.casefold()
