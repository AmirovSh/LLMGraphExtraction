from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from config.projection_identifiers import (
    DEFAULT_PROJECTION_TYPES,
    ProjectionIdentifier,
    ProjectionTypeNames,
)
from config.settings import load_project_settings
from runtime.json_neo4j_parity import (
    build_parity_diff,
    neo4j_relationship_properties,
    replace_namespace_transactionally,
)
from runtime.production_runner import run
from runtime.projection_manifest import (
    load_projection_manifest,
    projection_manifest,
    require_matching_projection,
)


ROOT = Path(__file__).resolve().parents[1]


def _configured(tmp_path: Path, *, entity_label: str, relation_type: str) -> Path:
    target = tmp_path / "config"
    shutil.copytree(ROOT / "config", target)
    path = target / "neo4j.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["graph_projection"] = {
        "entity_label": entity_label,
        "relation_type": relation_type,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


def test_projection_type_defaults_and_valid_custom_values(tmp_path: Path) -> None:
    defaults = load_project_settings(ROOT / "config").graph_projection.type_names
    assert defaults == DEFAULT_PROJECTION_TYPES
    assert defaults.as_dict() == {
        "entity_label": "FACT_ENTITY",
        "relation_type": "FACT_RELATION",
    }
    custom = load_project_settings(
        _configured(
            tmp_path,
            entity_label="LEGAL_ENTITY",
            relation_type="DOCUMENT_FACT",
        )
    ).graph_projection.type_names
    assert custom.as_dict() == {
        "entity_label": "LEGAL_ENTITY",
        "relation_type": "DOCUMENT_FACT",
    }


@pytest.mark.parametrize(
    "invalid",
    [
        "fact_entity",
        "Fact Entity",
        "fact-relation",
        "`MATCH (n) DETACH DELETE n`",
        "123_ENTITY",
        "ENTITY;",
        "A" * 65,
    ],
)
def test_invalid_projection_identifier_fails_before_side_effects(
    tmp_path: Path, invalid: str,
) -> None:
    config_dir = _configured(
        tmp_path, entity_label=invalid, relation_type="FACT_RELATION"
    )
    source = tmp_path / "source.txt"
    source.write_text("A synthetic source.", encoding="utf-8")
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="projection identifier"):
        run(
            source,
            output,
            config_dir,
            llm_post=lambda *_a, **_k: pytest.fail("transport reached"),
            embedding_resolver=lambda *_a, **_k: pytest.fail("embedding reached"),
            neo4j_importer=lambda *_a, **_k: pytest.fail("Neo4j reached"),
        )
    assert not output.exists()


class _Result(list):
    def consume(self) -> None:
        return None


class _CaptureSession:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def execute_write(self, callback) -> None:
        session = self

        class Tx:
            def run(self, query: str, **_params):
                session.queries.append(query)
                return _Result()

        callback(Tx())


class _ExistingProjectionSession:
    def execute_write(self, callback) -> None:
        class Tx:
            def run(self, query: str, **_params):
                if "RETURN DISTINCT n.run_id" in query:
                    return _Result([{
                        "run_id": "custom",
                        "source_sha256": "a" * 64,
                        "manifest_identity_hash": "b" * 64,
                        "labels": ["FACT_ENTITY"],
                        "relationship_types": ["FACT_RELATION"],
                    }])
                pytest.fail("namespace mutation occurred after projection mismatch")

        callback(Tx())


def _semantic_graph() -> tuple[list[dict], list[dict]]:
    nodes = [
        {
            "entity_id": "E001",
            "canonical_name": "Harbor Hub",
            "primary_type": "COMPONENT",
            "aliases": ["hub"],
            "local_entity_ids": ["M001", "M002"],
            "evidence_span_ids": ["S001"],
        },
        {
            "entity_id": "E002",
            "canonical_name": "Inspection Package",
            "primary_type": "DOCUMENT",
            "aliases": [],
            "local_entity_ids": ["M003"],
            "evidence_span_ids": ["S001"],
        },
    ]
    edges = [{
        "edge_id": "R001",
        "source_entity_id": "E001",
        "target_entity_id": "E002",
        "raw_relation": "coordinates",
        "relation_family": "ACTION",
        "relation_description": "The hub coordinates the package.",
        "evidence_span_ids": ["S001"],
        "support_level": "EXPLICIT",
        "qualifiers": {
            "temporality": {
                "surface": "from September 2026",
                "normalized": None,
                "evidence_span_ids": ["S001"],
            },
            "condition": None,
            "modality": "ASSERTED",
            "negated": False,
            "quantity": None,
            "version": None,
        },
        "unit_id": "U001",
        "provenance": {"source": "fact_extraction"},
    }]
    return nodes, edges


def test_custom_projection_queries_and_semantics_are_independent() -> None:
    projection = ProjectionTypeNames.parse(
        entity_label="LEGAL_ENTITY", relation_type="DOCUMENT_FACT"
    )
    nodes, edges = _semantic_graph()
    original = deepcopy((nodes, edges))
    session = _CaptureSession()
    replace_namespace_transactionally(
        session,
        graph_id="fact_extraction_custom",
        nodes=nodes,
        edges=edges,
        ownership={
            "run_id": "custom",
            "source_sha256": "a" * 64,
            "manifest_identity_hash": "b" * 64,
        },
        projection=projection,
    )
    rendered = "\n".join(session.queries)
    assert ":LEGAL_ENTITY" in rendered
    assert ":DOCUMENT_FACT" in rendered
    assert (nodes, edges) == original
    assert edges[0]["raw_relation"] == "coordinates"
    assert edges[0]["relation_family"] == "ACTION"
    assert edges[0]["evidence_span_ids"] == ["S001"]
    assert edges[0]["qualifiers"]["temporality"]["surface"] == "from September 2026"
    properties = neo4j_relationship_properties(edges[0], "fact_extraction_custom")
    assert properties["raw_relation"] == "coordinates"
    assert properties["relation_family"] == "ACTION"


def test_existing_namespace_with_different_projection_is_rejected_before_delete() -> None:
    nodes, edges = _semantic_graph()
    with pytest.raises(RuntimeError, match="different projection type names"):
        replace_namespace_transactionally(
            _ExistingProjectionSession(),
            graph_id="fact_extraction_custom",
            nodes=nodes,
            edges=edges,
            ownership={
                "run_id": "custom",
                "source_sha256": "a" * 64,
                "manifest_identity_hash": "b" * 64,
            },
            projection=ProjectionTypeNames.parse(
                entity_label="LEGAL_ENTITY", relation_type="DOCUMENT_RELATION"
            ),
        )


def test_parity_requires_configured_labels_types_and_indexes() -> None:
    projection = ProjectionTypeNames.parse(
        entity_label="LEGAL_ENTITY", relation_type="DOCUMENT_FACT"
    )
    nodes, edges = _semantic_graph()
    base = {
        "graph_id": "fact_extraction_custom",
        "json_nodes": nodes,
        "json_edges": edges,
        "neo4j_node_ids": {"E001", "E002"},
        "neo4j_edges": {"R001": ("E001", "E002")},
        "edges_without_graph_id": [],
        "duplicate_edges": [],
        "import_errors": [],
        "active_project_labels": {"LEGAL_ENTITY"},
        "active_project_relationship_types": {"DOCUMENT_FACT"},
        "project_indexes": [
            {"name": "legal_entity_node_graph_id", "labelsOrTypes": ["LEGAL_ENTITY"]},
            {"name": "document_fact_relationship_graph_id", "labelsOrTypes": ["DOCUMENT_FACT"]},
        ],
        "projection": projection,
    }
    assert build_parity_diff(**base)["status"] == "passed"
    mismatch = build_parity_diff(
        **(base | {"active_project_labels": {"FACT_ENTITY"}})
    )
    assert mismatch["status"] == "failed"
    assert mismatch["schema_parity"]["unexpected_project_labels"] == ["FACT_ENTITY"]


def test_projection_manifest_binds_rebuild_configuration(tmp_path: Path) -> None:
    recorded = ProjectionTypeNames.parse(
        entity_label="LEGAL_ENTITY", relation_type="DOCUMENT_FACT"
    )
    path = tmp_path / "projection_manifest.json"
    path.write_text(
        json.dumps(
            projection_manifest(
                graph_id="fact_extraction_custom", projection=recorded
            )
        ),
        encoding="utf-8",
    )
    loaded = load_projection_manifest(path, graph_id="fact_extraction_custom")
    require_matching_projection(recorded=loaded, configured=recorded)
    with pytest.raises(ValueError, match="differ from the run"):
        require_matching_projection(
            recorded=loaded, configured=DEFAULT_PROJECTION_TYPES
        )


def test_identifier_is_not_silently_normalized() -> None:
    with pytest.raises(ValueError, match="projection identifier"):
        ProjectionIdentifier("fact_entity")
