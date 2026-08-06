from __future__ import annotations

import pytest

from prompts.contracts import FactExtraction
from runtime.production_projection import project_extraction_unit
PROMPT_IDENTITY = {
    "prompt_id": "prompt_kimi_default",
}


def extraction_with_version_reference() -> FactExtraction:
    return FactExtraction.model_validate({
        "entities":[
            {"local_id":"M001","name":"Policy","type":"POLICY","evidence_span_ids":["S001"]},
            {"local_id":"M002","name":"Version 2.5","type":"VERSION","evidence_span_ids":["S001"]},
        ],
        "relations":[{
            "source_local_id":"M001","raw_relation":"applies_in_version","relation_family":"LIFECYCLE_VERSION",
            "relation_description":"Policy applies in version 2.5.","target_local_id":"M002","evidence_span_ids":["S001"],
            "support_level":"EXPLICIT","qualifiers":{"temporality":None,"condition":None,"modality":"ASSERTED",
            "negated":False,"quantity":None,"version":"M002"},
        }],
    })


def test_qualifier_local_id_is_rebound_to_correct_global_entity() -> None:
    graph=project_extraction_unit(unit_id="U002",extraction=extraction_with_version_reference(),
        resolved_entity_ids={"M001":"E_POLICY","M002":"E_VERSION"},valid_span_ids={"S001"},
        prompt_identity=PROMPT_IDENTITY)
    assert graph["edges"][0]["qualifiers"]["version"] == "E_VERSION"
    assert graph["edges"][0]["qualifiers"]["version"] != "E_POLICY"
    assert graph["edges"][0]["edge_id"] == "R_2182522958e1b32b"
    assert graph["edges"][0]["provenance"]["prompt_id"] == "prompt_kimi_default"


def test_final_projection_rejects_any_unresolved_local_id_in_qualifiers() -> None:
    extraction=extraction_with_version_reference(); payload=extraction.model_dump(mode="json")
    payload["relations"][0]["qualifiers"]["version"]="release M999"
    with pytest.raises(ValueError,match="unresolved local entity ID"):
        project_extraction_unit(unit_id="U001",extraction=FactExtraction.model_validate(payload),
            resolved_entity_ids={"M001":"E_POLICY","M002":"E_VERSION"},valid_span_ids={"S001"},
            prompt_identity=PROMPT_IDENTITY)
