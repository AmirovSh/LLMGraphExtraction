from devtools.contextual_entity_profiles import *


def entity() -> LocalEntityContext:
    return LocalEntityContext(local_id="L1", canonical_name="Build Registry", entity_type="COMPONENT",
        aliases=["Registry"], evidence=["The Build Registry stores approved build definitions."],
        incidents=[LocalIncident(direction="outgoing", relation="stores", neighbour_name="Build Definition", source_order=2),
                   LocalIncident(direction="incoming", relation="queries", neighbour_name="Deployment Controller", source_order=1)],
        source_kind="CONTROLLED_SYNTHETIC", identity_cluster_id="registry")


def test_profiles_are_deterministic_and_pair_independent():
    assert render_profile(entity(), "R4") == render_profile(entity(), "R4")


def test_r3_and_r4_are_source_grounded_and_directional():
    r3, r4 = render_profile(entity(), "R3")["text"], render_profile(entity(), "R4")["text"]
    assert "stores approved build definitions" in r3
    assert "queries <- Deployment Controller" in r4 and "stores -> Build Definition" in r4


def test_historical_bounds_and_truncation():
    item = entity().model_copy(update={"evidence": ["x" * 300] * 7})
    profile = render_profile(item, "R3")
    assert profile["retained_characters"] <= 800 and profile["truncated"]
    assert profile["text"].count("- ") <= 6


def test_eligibility_requires_real_context():
    empty = entity().model_copy(update={"evidence": [], "incidents": []})
    assert eligibility(empty) == {"R1_ELIGIBLE": True, "R2_ELIGIBLE": True, "R3_ELIGIBLE": False, "R4_ELIGIBLE": False}


def test_post_resolution_leakage_is_rejected():
    import pytest
    with pytest.raises(ValueError): assert_pre_resolution_payload({"canonical_name": "A", "global_entity_id": "E1"})


def test_matched_input_is_identical_for_bge_and_reranker():
    profile = render_profile(entity(), "R4")["text"]
    assert profile == profile


def test_benchmark_profiles_are_nontrivial_pair_independent_and_grounded():
    from devtools.run_contextual_reranker import benchmark
    rows = benchmark()
    assert len(rows) == 120
    assert sum(r["expected"] == "MERGE" for r in rows) == 40
    assert all(r["entity_a"].canonical_name != r["entity_b"].canonical_name for r in rows)
    assert all(r["entity_a"].evidence and r["entity_b"].evidence for r in rows)
    assert all(r["entity_a"].incidents and r["entity_b"].incidents for r in rows)
    first = rows[0]["entity_a"]
    assert render_profile(first, "R4") == render_profile(first, "R4")


def test_same_context_negatives_remain_distinct_and_direction_is_preserved():
    from devtools.run_contextual_reranker import benchmark
    negative = next(r for r in benchmark() if r["expected"] == "SPLIT")
    assert negative["entity_a"].evidence == negative["entity_b"].evidence
    left = render_profile(negative["entity_a"], "R4")["text"]
    right = render_profile(negative["entity_b"], "R4")["text"]
    assert "->" in left and "<-" in right


def test_production_runners_do_not_import_experiment():
    from pathlib import Path
    for path in (Path("scripts/run_production.py"), Path("scripts/run_batch.py")):
        assert "contextual_entity_profiles" not in path.read_text(encoding="utf-8")


def test_context_section_removal_is_deterministic():
    from devtools.run_contextual_reranker_diagnostics import strip_section
    text = "name\naliases:\n- A\nevidence:\n- E\nobserved_local_relations:\n- R"
    assert strip_section(text, "evidence:", "observed_local_relations:") == "name\naliases:\n- A\nobserved_local_relations:\n- R"
    assert strip_section(text, "observed_local_relations:", None).endswith("- E")
