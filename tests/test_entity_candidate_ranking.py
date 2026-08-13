import pytest
from devtools.entity_candidate_ranking import *


def entity(i,name,group="g"):
    return RankingEntity(entity_id=i,canonical_name=name,entity_type="ORGANIZATION",graph_group=group)


def test_generic_normalization_and_acronym_are_observable_not_semantic_maps():
    assert normalize_name("A.B-C") == "a b c"
    assert acronym_match("IBM","International Business Machines")
    # Structural generation intentionally admits ambiguous expansions; it never decides identity.
    assert acronym_match("IBM","International Backup Module")


def test_generators_never_cross_graph_groups():
    assert candidate_sources(entity("a","Same","a"),entity("b","Same","b"),1)==[]


def test_rank_metrics_support_multiple_positives():
    m=rank_metrics([[('a',1),('b',0),('c',1)]])
    assert m["recall_at_1"]==.5 and m["recall_at_3"]==1 and m["mrr"]==1


def test_rank_metrics_report_zero_generator_coverage_without_crashing():
    m=rank_metrics([[('a',0)]])
    assert m["queries"]==0 and m["recall_at_5"]==0 and m["median_true_match_rank"] is None


def test_neutral_tie_break_is_label_independent_and_stable():
    from devtools.run_reranker_candidate_ranking import neutral_key
    assert neutral_key("q","a")==neutral_key("q","a")
    assert neutral_key("q","a")!=neutral_key("q","b")


def test_ranked_candidate_contract_is_non_authoritative():
    item=RankedCandidate(query_entity_id="q",candidate_entity_id="c",rank=1,relevance_score=.9,candidate_sources=["BGE"])
    ensure_non_authoritative([item]); assert "merge" not in str(item.model_dump()).casefold()
    with pytest.raises(Exception): RankedCandidate(**item.model_dump(),should_merge=True)


def test_production_runners_do_not_import_candidate_ranker():
    from pathlib import Path
    for p in (Path("runtime/production_runner.py"),Path("runtime/batch_runner.py")):
        assert "entity_candidate_ranking" not in p.read_text(encoding="utf-8")
    config=Path("config/entity_resolution.yaml").read_text(encoding="utf-8")
    assert "candidate_similarity: 0.76" in config and "automatic_merge_similarity: 0.90" in config
    assert "input_format: canonical_name_only" in config and "top_k: 10" in config
