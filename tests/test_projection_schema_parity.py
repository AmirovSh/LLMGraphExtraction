from __future__ import annotations

import json
from copy import deepcopy

import pytest

from runtime.json_neo4j_parity import (
    EXPECTED_NODE_LABEL,
    EXPECTED_RELATIONSHIP_TYPE,
    NamespaceOwnershipError,
    build_parity_diff,
    neo4j_relationship_properties,
    replace_namespace_transactionally,
)


class _Result(list):
    def consume(self):
        return None


class _TransactionalSession:
    def __init__(self, store: dict, fail_at: str | None = None) -> None:
        self.store = store
        self.fail_at = fail_at

    def execute_write(self, callback) -> None:
        working = deepcopy(self.store)
        session = self

        class Tx:
            def run(self, query: str, **params):
                if "RETURN DISTINCT n.run_id" in query:
                    owner = working.get("owner")
                    return _Result([owner] if owner else [])
                if "DETACH DELETE" in query:
                    working["nodes"] = []
                    working["edges"] = []
                    if session.fail_at == "after_delete":
                        raise RuntimeError("injected after delete")
                    return _Result()
                if f"CREATE (n:{EXPECTED_NODE_LABEL})" in query:
                    working["nodes"] = deepcopy(params["rows"])
                    working["owner"] = {
                        key: params["rows"][0]["props"][key]
                        for key in ("run_id", "source_sha256", "manifest_identity_hash")
                    }
                    if session.fail_at == "after_nodes":
                        raise RuntimeError("injected after nodes")
                    return _Result()
                if f"CREATE (a)-[r:{EXPECTED_RELATIONSHIP_TYPE}]" in query:
                    working["edges"] = deepcopy(params["rows"])
                    if session.fail_at == "after_relationships":
                        raise RuntimeError("injected after relationships")
                    return _Result()
                raise AssertionError(query)

        callback(Tx())
        if self.fail_at == "before_commit":
            raise RuntimeError("injected before commit")
        self.store.clear()
        self.store.update(working)


def projection_payload() -> tuple[str, list[dict], list[dict]]:
    graph_id = "fact_extraction_projection"
    nodes = [
        {
            "entity_id": "E001",
            "canonical_name": "Policy",
            "primary_type": "CONTROL",
            "aliases": [],
            "local_entity_ids": ["M001"],
            "evidence_span_ids": ["S001"],
        },
        {
            "entity_id": "E002",
            "canonical_name": "Publication",
            "primary_type": "OUTCOME",
            "aliases": [],
            "local_entity_ids": ["M002"],
            "evidence_span_ids": ["S001"],
        },
    ]
    edges = [{
        "edge_id": "R001",
        "source_entity_id": "E001",
        "target_entity_id": "E002",
        "raw_relation": "prevents",
        "relation_family": "CONDITION_REQUIREMENT",
        "relation_description": "The policy prevents publication.",
        "evidence_span_ids": ["S001"],
        "support_level": "EXPLICIT",
        "qualifiers": {
            "temporality": None, "condition": None, "modality": "ASSERTED",
            "negated": False, "quantity": None, "version": None,
        },
        "unit_id": "U001",
        "provenance": {
            "prompt_id": "prompt_test", "source": "fact_extraction",
        },
    }]
    return graph_id, nodes, edges


def parity(**overrides) -> dict:
    graph_id, nodes, edges = projection_payload()
    ownership = {
        "run_id": "projection", "source_sha256": "a" * 64,
        "manifest_identity_hash": "b" * 64,
    }
    node_properties = {
        node["entity_id"]: {
            **node, "graph_id": graph_id, "display_name": node["canonical_name"],
            **ownership,
        }
        for node in nodes
    }
    edge_properties = {
        edge["edge_id"]: neo4j_relationship_properties(edge, graph_id, ownership)
        for edge in edges
    }
    arguments = {
        "graph_id": graph_id,
        "json_nodes": nodes,
        "json_edges": edges,
        "neo4j_node_ids": {"E001", "E002"},
        "neo4j_edges": {"R001": ("E001", "E002")},
        "edges_without_graph_id": [],
        "duplicate_edges": [],
        "import_errors": [],
        "neo4j_node_properties": node_properties,
        "neo4j_edge_properties": edge_properties,
        "active_project_labels": {EXPECTED_NODE_LABEL},
        "active_project_relationship_types": {EXPECTED_RELATIONSHIP_TYPE},
        "project_indexes": [
            {"name": "fact_entity_graph_id", "labelsOrTypes": ["FACT_ENTITY"]},
            {"name": "fact_relation_graph_id", "labelsOrTypes": ["FACT_RELATION"]},
        ],
        "ownership": ownership,
    }
    arguments.update(overrides)
    return build_parity_diff(**arguments)


def test_projection_contract_reports_data_schema_and_namespace_parity() -> None:
    result = parity()
    assert result["status"] == "passed"
    assert set(result) >= {"data_parity", "schema_parity", "namespace_parity"}
    assert not result["schema_parity"]["node_property_mismatches"]


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"active_project_labels": {"FACT_ENTITY", "OTHER_ENTITY"}}, "unexpected_project_labels"),
        (
            {"active_project_relationship_types": {"FACT_RELATION", "OTHER_RELATION"}},
            "unexpected_project_relationship_types",
        ),
        (
            {"project_indexes": [
                {"name": "fact_entity_graph_id", "labelsOrTypes": ["FACT_ENTITY"]},
                {"name": "fact_relation_graph_id", "labelsOrTypes": ["FACT_RELATION"]},
                {"name": "unexpected_entity_index", "labelsOrTypes": ["FACT_ENTITY"]},
            ]},
            "unexpected_project_indexes",
        ),
    ],
)
def test_projection_schema_rejects_unexpected_tokens(overrides, field: str) -> None:
    result = parity(**overrides)
    assert result["status"] == "failed"
    assert result["schema_parity"][field]


def test_projection_schema_rejects_missing_required_property() -> None:
    graph_id, nodes, _ = projection_payload()
    ownership = {
        "run_id": "projection", "source_sha256": "a" * 64,
        "manifest_identity_hash": "b" * 64,
    }
    properties = {
        node["entity_id"]: {
            **node, "graph_id": graph_id, "display_name": node["canonical_name"],
            **ownership,
        }
        for node in nodes
    }
    properties["E001"].pop("canonical_name")
    result = parity(neo4j_node_properties=properties)
    assert result["status"] == "failed"
    assert result["schema_parity"]["missing_required_node_properties"]["E001"] == [
        "canonical_name"
    ]


def test_projection_schema_rejects_contract_property_value_mismatch() -> None:
    graph_id, nodes, _ = projection_payload()
    ownership = {
        "run_id": "projection", "source_sha256": "a" * 64,
        "manifest_identity_hash": "b" * 64,
    }
    properties = {
        node["entity_id"]: {
            **node, "graph_id": graph_id, "display_name": node["canonical_name"],
            **ownership,
        }
        for node in nodes
    }
    properties["E001"]["canonical_name"] = "Different"
    result = parity(neo4j_node_properties=properties)
    assert result["status"] == "failed"
    assert result["schema_parity"]["node_property_mismatches"][0]["property"] == "canonical_name"


def test_projection_namespace_rejects_missing_or_invalid_graph_ids() -> None:
    result = parity(
        edges_without_graph_id=[{"relationship_id": "unscoped", "relationship_type": "FACT_RELATION"}],
        unexpected_graph_ids=[""],
    )
    assert result["status"] == "failed"
    assert result["namespace_parity"]["edges_without_graph_id"]
    assert result["namespace_parity"]["unexpected_graph_ids"] == [""]


@pytest.mark.parametrize(
    "failure", ["after_delete", "after_nodes", "after_relationships", "before_commit"],
)
def test_transactional_namespace_replacement_rolls_back_every_injected_failure(
    failure: str,
) -> None:
    graph_id, nodes, edges = projection_payload()
    owner = {
        "run_id": "run-one", "source_sha256": "a" * 64,
        "manifest_identity_hash": "b" * 64,
    }
    original = {"owner": deepcopy(owner), "nodes": ["old-node"], "edges": ["old-edge"]}
    store = deepcopy(original)
    with pytest.raises(RuntimeError, match="injected"):
        replace_namespace_transactionally(
            _TransactionalSession(store, failure), graph_id=graph_id,
            nodes=nodes, edges=edges, ownership=owner,
        )
    assert store == original


def test_transactional_namespace_replacement_rejects_different_owner_before_delete() -> None:
    graph_id, nodes, edges = projection_payload()
    existing = {
        "run_id": "other", "source_sha256": "c" * 64,
        "manifest_identity_hash": "d" * 64,
    }
    requested = {
        "run_id": "requested", "source_sha256": "a" * 64,
        "manifest_identity_hash": "b" * 64,
    }
    original = {"owner": existing, "nodes": ["old-node"], "edges": ["old-edge"]}
    store = deepcopy(original)
    with pytest.raises(NamespaceOwnershipError, match="different run identity"):
        replace_namespace_transactionally(
            _TransactionalSession(store), graph_id=graph_id,
            nodes=nodes, edges=edges, ownership=requested,
        )
    assert store == original


def test_transactional_namespace_replacement_batches_and_persists_owner() -> None:
    graph_id, nodes, edges = projection_payload()
    owner = {
        "run_id": "run-one", "source_sha256": "a" * 64,
        "manifest_identity_hash": "b" * 64,
    }
    store = {"owner": deepcopy(owner), "nodes": ["old-node"], "edges": ["old-edge"]}
    replace_namespace_transactionally(
        _TransactionalSession(store), graph_id=graph_id,
        nodes=nodes, edges=edges, ownership=owner,
    )
    assert len(store["nodes"]) == len(nodes)
    assert len(store["edges"]) == len(edges)
    assert store["owner"] == owner
