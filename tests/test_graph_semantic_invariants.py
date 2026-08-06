from __future__ import annotations

from typing import Any

import pytest

from prompts.contracts import FactExtraction
from runtime.production_projection import project_extraction_unit
from runtime.run_reports import isolated_node_ids

PROMPT_IDENTITY = {
    "prompt_id": "prompt_test",
}


def extraction(
    text_relation: str, *, negated: bool, source: str = "Condition Review",
    target: str = "Scheduling", family: str = "CONDITION_REQUIREMENT",
    temporality: str | None = None, evidence_text: str | None = None,
) -> FactExtraction:
    return FactExtraction.model_validate({
        "entities": [
            {
                "local_id": "M001", "name": source, "type": "CONTROL",
                "evidence_span_ids": ["S001"],
            },
            {
                "local_id": "M002", "name": target, "type": "OUTCOME",
                "evidence_span_ids": ["S001"],
            },
        ],
        "relations": [{
            "source_local_id": "M001",
            "raw_relation": text_relation,
            "relation_family": family,
            "relation_description": evidence_text or f"{source} {text_relation} {target}.",
            "target_local_id": "M002",
            "evidence_span_ids": ["S001"],
            "support_level": "EXPLICIT",
            "qualifiers": {
                "temporality": temporality,
                "condition": None,
                "modality": "ASSERTED",
                "negated": negated,
                "quantity": None,
                "version": None,
            },
        }],
    })


def projected_edges(value: FactExtraction) -> list[dict[str, Any]]:
    return project_extraction_unit(
        unit_id="U001",
        extraction=value,
        resolved_entity_ids={"M001": "E001", "M002": "E002", "M003": "E003"},
        valid_span_ids={"S001"},
        prompt_identity=PROMPT_IDENTITY,
    )["edges"]


@pytest.mark.parametrize(
    ("statement", "relation", "negated", "source", "target"),
    [
        ("Condition Review blocks scheduling.", "blocks", False, "Condition Review", "Scheduling"),
        ("Condition Review does not block scheduling.", "blocks", True, "Condition Review", "Scheduling"),
        ("The policy prevents publication.", "prevents", False, "Policy", "Publication"),
        ("The policy does not prevent publication.", "prevents", True, "Policy", "Publication"),
        ("The validator rejects invalid packages.", "rejects", False, "Validator", "Invalid packages"),
    ],
)
def test_negation_semantics_survive_extraction_validation_and_projection(
    statement: str, relation: str, negated: bool, source: str, target: str,
) -> None:
    value = extraction(
        relation, negated=negated, source=source, target=target,
        evidence_text=statement,
    )
    edge = projected_edges(value)[0]
    assert [entity.name for entity in value.entities] == [source, target]
    assert value.relations[0].relation_description == statement
    assert edge["raw_relation"] == relation
    assert edge["qualifiers"]["negated"] is negated
    assert edge["source_entity_id"] == "E001"
    assert edge["target_entity_id"] == "E002"


@pytest.mark.parametrize(
    ("statement", "relation", "source", "target", "temporality"),
    [
        (
            "Version 3.6 was active before Version 3.7.",
            "was active before", "Version 3.6", "Version 3.7", "before Version 3.7",
        ),
        (
            "Version 3.7 followed Version 3.6.",
            "followed", "Version 3.7", "Version 3.6", "after Version 3.6",
        ),
        (
            "Version 3.6 remained active until Version 3.7 was introduced.",
            "remained active until", "Version 3.6", "Version 3.7",
            "until Version 3.7 was introduced",
        ),
    ],
)
def test_temporal_order_is_preserved_without_synthetic_replacement(
    statement: str, relation: str, source: str, target: str, temporality: str,
) -> None:
    edge = projected_edges(extraction(
        relation, negated=False, source=source, target=target,
        family="PROCESSING_SEQUENCE", temporality=temporality,
        evidence_text=statement,
    ))[0]
    assert edge["raw_relation"] != "replaces"
    assert edge["qualifiers"]["temporality"] == temporality
    assert edge["source_entity_id"] != edge["target_entity_id"]


@pytest.mark.parametrize(
    ("statement", "source", "target"),
    [
        ("Version 3.7 replaced Version 3.6.", "Version 3.7", "Version 3.6"),
        ("Version 3.6 was superseded by Version 3.7.", "Version 3.7", "Version 3.6"),
    ],
)
def test_explicit_replacement_preserves_new_to_old_direction(
    statement: str, source: str, target: str,
) -> None:
    edge = projected_edges(extraction(
        "replaces", negated=False, source=source, target=target,
        family="LIFECYCLE_VERSION", evidence_text=statement,
    ))[0]
    assert (edge["source_entity_id"], edge["raw_relation"], edge["target_entity_id"]) == (
        "E001", "replaces", "E002",
    )


def test_evidence_backed_isolated_entity_does_not_receive_synthetic_edge() -> None:
    value = FactExtraction.model_validate({
        "entities": [
            {
                "local_id": "M001", "name": "Registry", "type": "COMPONENT",
                "evidence_span_ids": ["S001"],
            },
            {
                "local_id": "M002", "name": "Record", "type": "RECORD",
                "evidence_span_ids": ["S001"],
            },
            {
                "local_id": "M003", "name": "Evidence Note", "type": "ASSERTION",
                "evidence_span_ids": ["S001"],
            },
        ],
        "relations": [{
            "source_local_id": "M001", "raw_relation": "stores",
            "relation_family": "DATA_FLOW", "relation_description": "Registry stores Record.",
            "target_local_id": "M002", "evidence_span_ids": ["S001"],
            "support_level": "EXPLICIT",
            "qualifiers": {
                "temporality": None, "condition": None, "modality": "ASSERTED",
                "negated": False, "quantity": None, "version": None,
            },
        }],
    })
    projection = project_extraction_unit(
        unit_id="U001", extraction=value,
        resolved_entity_ids={"M001": "E001", "M002": "E002", "M003": "E003"},
        valid_span_ids={"S001"}, prompt_identity=PROMPT_IDENTITY,
    )
    edge = projection["edges"][0]
    assert len(projection["edges"]) == 1
    assert edge["provenance"]["source"] == "fact_extraction"
    assert edge["evidence_span_ids"] == ["S001"]
    nodes = [{"entity_id": f"E{index:03d}"} for index in range(1, 4)]
    assert isolated_node_ids(nodes, projection["edges"]) == ["E003"]


def test_runtime_has_no_forbidden_synthetic_edge_stage() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    runtime_text = "\n".join(
        path.read_text(encoding="utf-8") for path in (root / "runtime").glob("*.py")
    ).lower()
    for forbidden in (
        "nearest_neighbor_edge", "semantic_repair_edge", "connect_graph_components",
    ):
        assert forbidden not in runtime_text
