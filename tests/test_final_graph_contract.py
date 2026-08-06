from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from runtime.final_graph_contract import validate_final_graph


def graph_payload() -> dict:
    provenance = {
        "prompt_id": "prompt_test",
        "source": "fact_extraction",
    }
    return {
        "graph_id": "fact_extraction_contract",
        "run_id": "contract",
        "prompt_id": "prompt_test",
        "prompt_hash": "a" * 64,
        "manifest_identity_hash": "b" * 64,
        "nodes": [
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
        ],
        "edges": [
            {
                "source_entity_id": "E001",
                "target_entity_id": "E002",
                "raw_relation": "prevents",
                "relation_family": "CONDITION_REQUIREMENT",
                "relation_description": "The policy prevents publication.",
                "evidence_span_ids": ["S001"],
                "support_level": "EXPLICIT",
                "qualifiers": {
                    "temporality": None,
                    "condition": None,
                    "modality": "ASSERTED",
                    "negated": False,
                    "quantity": None,
                    "version": None,
                },
                "unit_id": "U001",
                "provenance": provenance,
                "edge_id": "R001",
            }
        ],
        "source": {
            "source_name": "sample_input.txt",
            "source_sha256": "c" * 64,
            "source_size_bytes": 42,
            "source_location_policy": "repository_relative",
            "repository_relative_path": "examples/sample_input.txt",
            "units": [{"unit_id": "U001", "span_ids": ["S001"]}],
        },
    }


def test_final_graph_round_trip_preserves_contract_values() -> None:
    payload = graph_payload()
    payload["edges"][0]["raw_relation"] = "  prevents exactly  "
    payload["nodes"][0]["canonical_name"] = " Policy — 4.2 "
    validated = validate_final_graph(payload).model_dump(mode="json")
    assert validated == payload


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda graph: graph["edges"][0].update(raw_relation=" \t "), "non-empty string"),
        (lambda graph: graph["nodes"].append(deepcopy(graph["nodes"][0])), "duplicate node ID"),
        (lambda graph: graph["edges"].append(deepcopy(graph["edges"][0])), "duplicate edge ID"),
        (lambda graph: graph["edges"][0].update(source_entity_id="E999"), "source endpoint"),
        (lambda graph: graph["edges"][0]["provenance"].update(prompt_id="other@1"), "prompt_id"),
        (lambda graph: graph["edges"][0]["qualifiers"].update(version="M010"), "unresolved local"),
        (lambda graph: graph["edges"][0].update(target_entity_id="M002"), "unresolved local"),
        (
            lambda graph: graph["edges"][0].update(evidence_span_ids=["S999"]),
            "evidence span",
        ),
        (lambda graph: graph["edges"][0].update(unit_id="U999"), "unit_id"),
    ],
)
def test_final_graph_rejects_invalid_authoritative_artifacts(mutation, message: str) -> None:
    payload = graph_payload()
    mutation(payload)
    with pytest.raises(ValidationError, match=message):
        validate_final_graph(payload)


def test_final_graph_rejects_unknown_fields_instead_of_dropping_them() -> None:
    payload = graph_payload()
    payload["nodes"][0]["unknown_required_field"] = "value"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate_final_graph(payload)
