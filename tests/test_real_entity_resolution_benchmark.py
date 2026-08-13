import pytest

from devtools.real_entity_resolution_benchmark import digest, freeze, split_clusters, validate


def fixtures():
    clusters=[{"identity_cluster_id":"I1","members":[{"entity_id":"a","graph_group":"g"},{"entity_id":"b","graph_group":"g"}]}]
    pairs=[{"left_entity_id":"a","right_entity_id":"b","label":"SAME_ENTITY","confidence":"HIGH","category":"SHORT_FULL","evidence_refs":["d:S1"]}]
    queries=[{"query_id":"q","query_entity_id":"a","candidate_pool":[{"candidate_entity_id":"b","expected":"SAME_ENTITY"}]}]
    return clusters,pairs,queries


def test_cluster_split_is_stable_and_isolated():
    clusters=[{"identity_cluster_id":f"I{i}","members":[]} for i in range(20)]
    assert split_clusters(clusters)==split_clusters(list(reversed(clusters)))
    assert set(split_clusters(clusters))=={f"I{i}" for i in range(20)}


def test_validation_rejects_score_dependency_and_cross_group():
    clusters,pairs,queries=fixtures(); pairs[0]["score"]=.9
    with pytest.raises(ValueError,match="model output"): validate(clusters,pairs,queries)
    clusters,pairs,queries=fixtures(); clusters.append({"identity_cluster_id":"I2","members":[{"entity_id":"c","graph_group":"x"}]}); queries[0]["candidate_pool"].append({"candidate_entity_id":"c"})
    with pytest.raises(ValueError,match="cross-group"): validate(clusters,pairs,queries)


def test_freeze_has_stable_hash_and_failed_quality_gate(tmp_path):
    clusters,pairs,queries=fixtures()
    one=freeze(tmp_path/"a",[{"document_id":"d"}],[{"local_entity_count":2}],clusters,pairs,[],queries,[],[])
    two=freeze(tmp_path/"b",[{"document_id":"d"}],[{"local_entity_count":2}],clusters,pairs,[],queries,[],[])
    assert one["manifest"]==two["manifest"]
    assert one["quality"]["status"]=="FAIL" and one["quality"]["stage_b"]=="NOT_RUN"


def test_duplicate_query_and_inconsistent_positive_are_rejected():
    clusters,pairs,queries=fixtures(); queries.append(dict(queries[0]))
    with pytest.raises(ValueError,match="duplicate query"): validate(clusters,pairs,queries)
    clusters,pairs,queries=fixtures(); clusters.append({"identity_cluster_id":"I2","members":[{"entity_id":"c","graph_group":"g"}]}); pairs[0]["right_entity_id"]="c"
    with pytest.raises(ValueError,match="share a cluster"): validate(clusters,pairs,queries)


def test_hash_is_order_key_stable():
    assert digest({"a":1,"b":2})==digest({"b":2,"a":1})


def test_source_manifest_requires_portable_references(tmp_path):
    clusters,pairs,queries=fixtures()
    source={"document_id":"d","path":"benchmark_sources/d.txt","sha256":"a"*64}
    result=freeze(tmp_path,[source],[],clusters,pairs,[],queries,[],[])
    assert result["statistics"]["total_source_documents"]==1
    assert not source["path"].startswith(("/","C:"))
