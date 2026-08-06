from __future__ import annotations

import importlib.util
from pathlib import Path

from config.settings import load_project_settings
from prompts.registry import resolve_prompt
from runtime.json_neo4j_parity import build_parity_diff
from runtime.observable_entity_resolution import evaluate_similarity
from runtime.production_inputs import sentence_spans
from runtime.run_reports import deduplicate_exact_edges, isolated_node_ids


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_production", ROOT / "scripts" / "run_production.py")
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _settings():
    return load_project_settings(ROOT / "config")


def test_sentence_split_preserves_dotted_structured_values() -> None:
    spans = sentence_spans("Version 3.2 uses 10.4.0.7 on 2026.07.21. Validation continues.")
    assert [span["text"] for span in spans] == [
        "Version 3.2 uses 10.4.0.7 on 2026.07.21.",
        "Validation continues.",
    ]


def test_fact_extraction_schema_uses_centralized_bounds() -> None:
    settings = _settings().extraction.dynamic
    schema = resolve_prompt("prompt_kimi_default").schema_builder(
        max_entities=settings.max_entities_per_unit,
        max_relations=settings.max_relations_per_unit,
        max_evidence_spans=settings.max_evidence_spans_per_item,
        max_relation_description_characters=settings.max_relation_description_characters,
    )
    assert schema["properties"]["entities"]["maxItems"] == 60
    assert schema["properties"]["relations"]["maxItems"] == 40
    assert "relation_description" not in str(schema)


def test_type_is_trace_only_and_not_a_merge_filter() -> None:
    entities = [
        {"local_id": "M001", "name": "Platform", "type": "COMPONENT", "unit_id": "U001"},
        {"local_id": "M002", "name": "platform", "type": "ROLE", "unit_id": "U002"},
    ]
    _, decisions = evaluate_similarity(
        entities, [[1.0, 0.0], [1.0, 0.0]], [], candidate_threshold=.76, merge_threshold=.90,
        decisions=_settings().entity_resolution.decisions.model_dump(),
    )
    assert decisions[0]["decision"] == "MERGE"
    assert decisions[0]["type_used_as_hard_filter"] is False


def test_exact_dedup_and_parity_diff_are_explicit() -> None:
    edge = {"edge_id": "R001", "source_entity_id": "E001", "target_entity_id": "E002"}
    edges, duplicates = deduplicate_exact_edges([edge, edge])
    diff = build_parity_diff(
        graph_id="test", json_nodes=[{"entity_id": "E001"}, {"entity_id": "E002"}], json_edges=edges,
        neo4j_node_ids={"E001", "E002"}, neo4j_edges={"R001": ("E001", "E002")},
        edges_without_graph_id=[], duplicate_edges=duplicates, import_errors=[],
    )
    assert diff["duplicate_edges"] == ["R001"]
    assert not diff["json_only_edges"]
    assert not diff["source_target_mismatches"]


def test_parity_diff_detects_endpoint_mismatch() -> None:
    diff = build_parity_diff(
        graph_id="test", json_nodes=[{"entity_id": "E001"}, {"entity_id": "E002"}],
        json_edges=[{"edge_id": "R001", "source_entity_id": "E001", "target_entity_id": "E002"}],
        neo4j_node_ids={"E001", "E002"}, neo4j_edges={"R001": ("E002", "E001")},
        edges_without_graph_id=[], duplicate_edges=[], import_errors=[],
    )
    assert diff["source_target_mismatches"][0]["edge_id"] == "R001"


def test_isolated_nodes_are_reported_without_creating_edges() -> None:
    nodes = [{"entity_id": "E001"}, {"entity_id": "E002"}, {"entity_id": "E003"}]
    edges = [{"source_entity_id": "E001", "target_entity_id": "E002"}]
    assert isolated_node_ids(nodes, edges) == ["E003"]


def test_relation_coverage_clarification_does_not_raise_schema_limit() -> None:
    settings = _settings().extraction.dynamic
    schema = resolve_prompt("prompt_kimi_default").schema_builder(
        max_entities=settings.max_entities_per_unit,
        max_relations=settings.max_relations_per_unit,
        max_evidence_spans=settings.max_evidence_spans_per_item,
        max_relation_description_characters=settings.max_relation_description_characters,
    )
    assert schema["properties"]["relations"]["maxItems"] == 40
