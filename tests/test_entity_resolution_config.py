from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from config.settings import load_project_settings, write_resolved_run_config
from runtime.observable_entity_resolution import evaluate_saved_matrix, evaluate_similarity, resolve_pair_decisions
from runtime.production_runner import run


ROOT = Path(__file__).resolve().parents[1]
DECISIONS = {
    "type_is_hard_filter": False,
    "require_context_for_semantic_alias_merge": True,
    "exact_name_cross_unit_merge": True,
    "case_only_name_merge": True,
    "semantic_conflict_blocks_merge": True,
}


def _entities() -> list[dict]:
    return [
        {"local_id": "M001", "name": "Platform", "type": "RESPONSIBILITY", "unit_id": "U001"},
        {"local_id": "M002", "name": "platform", "type": "ROLE", "unit_id": "U002"},
    ]


def test_type_mismatch_alone_never_forces_keep_separate() -> None:
    _, decisions = evaluate_similarity(_entities(), [[1.0, 0.0], [1.0, 0.0]], [], candidate_threshold=.76, merge_threshold=.90, decisions=DECISIONS)
    assert decisions[0]["decision"] == "MERGE"
    assert decisions[0]["type_match"] is False
    assert decisions[0]["type_used_as_hard_filter"] is False


def test_type_match_alone_never_permits_merge() -> None:
    entities = [
        {"local_id": "M001", "name": "A", "type": "COMPONENT", "unit_id": "U001"},
        {"local_id": "M002", "name": "B", "type": "COMPONENT", "unit_id": "U001"},
    ]
    _, decisions = evaluate_similarity(entities, [[1.0, 0.0], [1.0, 0.0]], [], candidate_threshold=.76, merge_threshold=.90, decisions=DECISIONS)
    assert decisions[0]["decision"] == "UNRESOLVED"


def test_case_only_name_merge_respects_its_own_setting() -> None:
    policy = {**DECISIONS, "case_only_name_merge": False}
    _, decisions = evaluate_similarity(
        _entities(), [[1.0, 0.0], [1.0, 0.0]], [],
        candidate_threshold=.76, merge_threshold=.90, decisions=policy,
    )
    assert decisions[0]["decision"] == "UNRESOLVED"


def test_typed_semantic_conflict_forces_keep_separate() -> None:
    entities = [
        {"local_id": "M001", "name": "X", "type": "STATE", "unit_id": "U001", "semantic_attributes": {"state": "A"}},
        {"local_id": "M002", "name": "X", "type": "STATE", "unit_id": "U002", "semantic_attributes": {"state": "B"}},
    ]
    _, decisions = evaluate_similarity(entities, [[1.0, 0.0], [1.0, 0.0]], [], candidate_threshold=.76, merge_threshold=.90, decisions=DECISIONS)
    assert decisions[0]["decision"] == "KEEP_SEPARATE"
    assert decisions[0]["semantic_conflict"] is True


def test_candidate_threshold_is_not_automatic_merge_threshold() -> None:
    _, decisions = evaluate_similarity(_entities(), [[1.0, 0.0], [.8, .6]], [], candidate_threshold=.76, merge_threshold=.90, decisions=DECISIONS)
    assert decisions[0]["decision"] == "UNRESOLVED"


@pytest.mark.parametrize(("left", "right"), [
    ("Build Registry", "Build Registry Record"),
    ("Package Registry", "Package Registry Record"),
])
def test_name_expansion_blocks_automatic_merge(left: str, right: str) -> None:
    entities=[{"local_id":"M001","name":left,"type":"COMPONENT","unit_id":"U001"},{"local_id":"M002","name":right,"type":"COMPONENT","unit_id":"U002"}]
    _, decisions=evaluate_similarity(entities,[[1.0,0.0],[1.0,0.0]],[],candidate_threshold=.76,merge_threshold=.90,decisions=DECISIONS)
    assert decisions[0]["decision"] == "UNRESOLVED"
    assert decisions[0]["name_relation"] == "TOKEN_CONTAINMENT_EXPANSION"
    assert decisions[0]["automatic_merge_blocked_by_name_expansion"] is True
    assert decisions[0]["type_used_as_hard_filter"] is False


def test_shared_token_without_containment_does_not_permit_merge() -> None:
    entities=[{"local_id":"M001","name":"Deployment Plan","type":"ARTIFACT","unit_id":"U001"},{"local_id":"M002","name":"Deployment Record","type":"ARTIFACT","unit_id":"U001"}]
    _, decisions=evaluate_similarity(entities,[[1.0,0.0],[1.0,0.0]],[],candidate_threshold=.76,merge_threshold=.90,decisions=DECISIONS)
    assert decisions[0]["decision"] == "UNRESOLVED" and decisions[0]["name_relation"] == "DIFFERENT"


def _relation(source: str, raw_relation: str, target: str, *, relation_id: str = "L001") -> dict:
    return {
        "relation_id": relation_id,
        "source_local_id": source,
        "raw_relation": raw_relation,
        "relation_family": "ASSERTION_CLAIM",
        "relation_description": "The source and target have the stated relation.",
        "target_local_id": target,
        "evidence_span_ids": ["S001"],
        "unit_id": "U001",
    }


@pytest.mark.parametrize(("left", "right"), [
    ("Harbor Operations Hub", "harbor hub"),
    ("Asset Catalog", "the catalog"),
])
def test_evidence_backed_explicit_equivalence_merges_below_embedding_threshold(left: str, right: str) -> None:
    entities = [
        {"local_id":"M001","name":left,"type":"COMPONENT","unit_id":"U001"},
        {"local_id":"M002","name":right,"type":"ALIAS","unit_id":"U001"},
    ]
    relations = [_relation("M001", "also_called", "M002")]
    _, decisions = evaluate_similarity(
        entities, [[1.0,0.0],[0.0,1.0]], relations,
        candidate_threshold=.76, merge_threshold=.90, decisions=DECISIONS,
    )
    decision = decisions[0]
    assert decision["decision"] == "MERGE"
    assert decision["explicit_equivalence_evidence"] is True
    assert decision["equivalence_relation_ids"] == ["L001"]
    assert decision["equivalence_evidence_span_ids"] == ["S001"]
    assert decision["merge_signal"] == "EXPLICIT_EQUIVALENCE"
    assert decision["type_used_as_hard_filter"] is False
    assert decision["automatic_merge_blocked_by_name_expansion"] is False
    ids, report = resolve_pair_decisions(entities, decisions)
    assert ids["M001"] == ids["M002"]
    assert report["merged_pairs"][0]["merge_signal"] == "EXPLICIT_EQUIVALENCE"


@pytest.mark.parametrize(("left", "relation", "right"), [
    ("Asset Catalog", "stores", "Asset Catalog Record"),
    ("Build Registry", "references", "Build Registry Record"),
    ("Version 2.5", "replaces", "Version 2.6"),
])
def test_ordinary_semantic_relation_never_becomes_explicit_equivalence(
    left: str, relation: str, right: str,
) -> None:
    entities = [
        {"local_id":"M001","name":left,"type":"COMPONENT","unit_id":"U001"},
        {"local_id":"M002","name":right,"type":"COMPONENT","unit_id":"U001"},
    ]
    _, decisions = evaluate_similarity(
        entities, [[1.0,0.0],[1.0,0.0]], [_relation("M001", relation, "M002")],
        candidate_threshold=.76, merge_threshold=.90, decisions=DECISIONS,
    )
    assert decisions[0]["decision"] != "MERGE"
    assert decisions[0]["explicit_equivalence_evidence"] is False
    assert decisions[0]["merge_signal"] is None


def test_explicit_equivalence_with_conflicting_stable_ids_does_not_merge() -> None:
    entities = [
        {"local_id":"M001","name":"Service Alias","type":"COMPONENT","unit_id":"U001","stable_identifier":"svc-1"},
        {"local_id":"M002","name":"service","type":"COMPONENT","unit_id":"U001","stable_identifier":"svc-2"},
    ]
    _, decisions = evaluate_similarity(
        entities, [[1.0,0.0],[0.0,1.0]], [_relation("M001", "same_as", "M002")],
        candidate_threshold=.76, merge_threshold=.90, decisions=DECISIONS,
    )
    assert decisions[0]["decision"] == "KEEP_SEPARATE"
    assert decisions[0]["explicit_equivalence_evidence"] is True
    assert decisions[0]["semantic_conflict_fields"] == ["stable_identifier"]


@pytest.mark.parametrize(("left", "right", "expected"), [
    ("Build Registry", "build registry", "CASE_ONLY"),
    ("Release Procedure", "release procedure", "CASE_ONLY"),
])
def test_case_only_equivalents_can_merge(left: str, right: str, expected: str) -> None:
    entities=[{"local_id":"M001","name":left,"type":"COMPONENT","unit_id":"U001"},{"local_id":"M002","name":right,"type":"ROLE","unit_id":"U002"}]
    _, decisions=evaluate_similarity(entities,[[1.0,0.0],[1.0,0.0]],[],candidate_threshold=.76,merge_threshold=.90,decisions=DECISIONS)
    assert decisions[0]["decision"] == "MERGE" and decisions[0]["name_relation"] == expected
    assert decisions[0]["automatic_merge_blocked_by_name_expansion"] is False


def test_saved_matrix_replay_admits_release_platform_candidate() -> None:
    entities = [
        {"local_id": "M001", "name": "Release Platform", "type": "COMPONENT", "unit_id": "U001"},
        {"local_id": "M002", "name": "release-management platform", "type": "COMPONENT", "unit_id": "U002"},
    ]
    saved = {"retrieval_method": "full_pairwise_cosine_matrix", "entity_local_ids": ["M001", "M002"], "matrix": [[1.0, .798407], [.798407, 1.0]]}
    _, decisions = evaluate_saved_matrix(entities, saved, [], candidate_threshold=.76, merge_threshold=.90, decisions=DECISIONS)
    assert decisions[0]["decision"] == "UNRESOLVED"


def test_settings_are_centralized_and_resolved_config_has_no_secret(tmp_path: Path) -> None:
    settings = load_project_settings(ROOT / "config")
    assert settings.entity_resolution.thresholds.candidate_similarity == .76
    assert settings.entity_resolution.thresholds.automatic_merge_similarity == .90
    target = tmp_path / "resolved_run_config.json"
    write_resolved_run_config(target, settings)
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert "test-secret-value" not in json.dumps(saved).lower()
    assert saved["entity_resolution"]["decisions"]["type_is_hard_filter"] is False
    assert set(saved) == {
        "extraction",
        "extraction_contracts",
        "entity_resolution",
        "runtime",
        "neo4j",
        "graph_projection",
    }
    assert "retries" not in saved["extraction"] and "concurrency" not in saved["extraction"]
    assert "observability" not in saved["entity_resolution"]
    assert "import_batch_size" not in saved["neo4j"]
    assert saved["entity_resolution"]["embedding"]["input_format"] == "canonical_name_only"
    assert saved["entity_resolution"]["retrieval"]["method_for_large_run"] == "top_k_from_full_matrix"
    remote = saved["runtime"]["remote"]
    assert remote["llm_provider_profile"] == "kimi_k2_6_vllm_structured"
    assert remote["provider_profiles"]["kimi_k2_6_vllm_instant"][
        "request_extra_body"
    ]["chat_template_kwargs"] == {
        "thinking": False, "preserve_thinking": False,
    }
    assert remote["capability_probe"]["plain_chat_max_output_tokens"] == 1024
    assert remote["sampling_profiles"]["kimi_instant_default"] == {
        "temperature": 0.6, "top_p": 0.95, "seed": None,
    }
    assert remote["sampling_profiles"]["kimi_structured_deterministic"] == {
        "temperature": 0.0, "top_p": None, "seed": None,
    }
    assert "test-secret-value" not in json.dumps(remote).casefold()
    assert remote["llm_api_key_env"] == "OPENAI_API_KEY"


def test_invalid_config_fails_before_model_call(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    shutil.copytree(ROOT / "config", config_dir)
    path = config_dir / "entity_resolution.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("candidate_similarity: 0.76", "candidate_similarity: 0.95"), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate_similarity"):
        load_project_settings(config_dir)


def test_runtime_module_has_no_embedded_production_threshold_constants() -> None:
    source = (ROOT / "runtime" / "observable_entity_resolution.py").read_text(encoding="utf-8")
    assert "0.82" not in source and "0.90" not in source


@pytest.mark.parametrize(("filename", "field_path", "value", "error_field"), [
    ("entity_resolution.yaml", ("entity_resolution", "enabled"), False, "enabled"),
    ("entity_resolution.yaml", ("entity_resolution", "embedding", "input_format"), "name_and_context", "input_format"),
    ("entity_resolution.yaml", ("entity_resolution", "retrieval", "method_for_small_run"), "approximate_ann", "method_for_small_run"),
    ("entity_resolution.yaml", ("entity_resolution", "retrieval", "method_for_large_run"), "top_k", "method_for_large_run"),
    ("entity_resolution.yaml", ("entity_resolution", "decisions", "type_is_hard_filter"), True, "type_is_hard_filter"),
    ("neo4j.yaml", ("neo4j", "namespace_property"), "tenant_id", "namespace_property"),
    ("extraction.yaml", ("extraction", "retries"), {"max_primary_retries": 1}, "retries"),
    ("extraction.yaml", ("extraction", "concurrency"), {"primary_extraction_workers": 2}, "concurrency"),
    ("entity_resolution.yaml", ("entity_resolution", "observability"), {"save_embedding_inputs": False}, "observability"),
    ("runtime.yaml", ("runtime", "remote", "llm_provider_profile"), "unknown", "llm_provider_profile"),
])
def test_unsupported_or_removed_config_fails_before_model_call(
    tmp_path: Path, filename: str, field_path: tuple[str, ...], value: object, error_field: str,
) -> None:
    config_dir = tmp_path / "config"; shutil.copytree(ROOT / "config", config_dir)
    path = config_dir / filename; payload = yaml.safe_load(path.read_text(encoding="utf-8")); target = payload
    for part in field_path[:-1]: target = target[part]
    target[field_path[-1]] = value; path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    input_path = tmp_path / "source.txt"; input_path.write_text("No call may occur.", encoding="utf-8"); called = False

    def forbidden_post(*args, **kwargs):
        nonlocal called; called = True; raise AssertionError("model call occurred")

    with pytest.raises(ValueError, match=error_field):
        run(input_path, tmp_path / "output", config_dir, llm_post=forbidden_post)
    assert called is False


def test_active_kimi_sampling_omits_seed_and_retired_seed_profile() -> None:
    remote = load_project_settings(ROOT / "config").runtime.remote
    assert remote.selected_provider_profile.sampling_profile == (
        "kimi_structured_temperature_zero"
    )
    assert remote.selected_provider_profile.temperature == 0
    assert remote.selected_provider_profile.top_p is None
    assert remote.selected_provider_profile.seed is None
    assert "kimi_structured_seed_42" not in remote.sampling_profiles
