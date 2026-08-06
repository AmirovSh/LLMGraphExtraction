from __future__ import annotations

from pathlib import Path
import json

import pytest
from pydantic import ValidationError

from config.settings import load_project_settings
from prompts.contracts import (
    SpanSemanticTemporalExtraction,
    build_tool_schema,
)
from prompts.registry import resolve_prompt
from runtime.production_inputs import render_prompt, request_payload
from runtime.span_extraction import (
    ExtractionContractViolation, aggregate_namespaced_results,
)
from runtime.span_semantic_temporal_contract import (
    validate_and_enrich_semantic_temporal,
)


ROOT = Path(__file__).resolve().parents[1]


def _payload(*, bindings: list[dict] | None = None) -> dict:
    return {
        "status": "facts_present",
        "entities": [
            {"canonical_name": "Policy", "entity_type": "CONTROL"},
            {"canonical_name": "applicability", "entity_type": "CONCEPT"},
        ],
        "relations": [{
            "source_entity_index": 0,
            "target_entity_index": 1,
            "relation_family": "CONDITION_REQUIREMENT",
            "raw_relation": "applies",
            "condition": None,
            "modality": "ASSERTED",
            "quantity": None,
            "version": None,
            "negated": False,
        }],
        "temporal_bindings": bindings or [],
        "no_graph_fact_reason": None,
    }


def _enrich(payload: dict, evidence: str = "Policy applies.") -> dict:
    return validate_and_enrich_semantic_temporal(
        SpanSemanticTemporalExtraction.model_validate(payload),
        unit_id="U001_S001", span_id="S001", evidence_text=evidence,
    )


def test_model_schema_contains_only_semantic_fields() -> None:
    schema = build_tool_schema(
        max_entities=60, max_relations=40, max_evidence_spans=10,
        max_relation_description_characters=300,
    )
    serialized = str(schema)
    for forbidden in (
        "unit_id", "span_id", "evidence_span_ids", "local_id",
        "relation_local_id", "graph_id", "namespace", "run_id",
    ):
        assert forbidden not in serialized
    relation = schema["$defs"]["SemanticRelation"]
    assert "relation_family" in relation["required"]
    assert set(schema["$defs"]["RelationFamily"]["enum"]) == {
        "COMPOSITION", "DATA_FLOW", "PROCESSING_SEQUENCE", "DEPENDENCY",
        "VALIDATION_QUALITY", "RESPONSIBILITY_OWNERSHIP",
        "LIFECYCLE_VERSION", "CONDITION_REQUIREMENT", "CAUSAL_RESULT",
        "ASSERTION_CONTEXT",
    }
    assert "temporal_bindings" in schema["required"]


def test_adapter_adds_stable_ids_and_request_evidence_only() -> None:
    result = _enrich(_payload())
    assert [item["local_id"] for item in result["entities"]] == [
        "S001_M001", "S001_M002",
    ]
    assert result["relations"][0]["relation_local_id"] == "S001_R001"
    assert result["relations"][0]["evidence_span_ids"] == ["S001"]
    assert result["relations"][0]["support_level"] == "EXPLICIT"
    aggregate = aggregate_namespaced_results([result])[0][1]
    assert aggregate.relations[0].raw_relation == "applies"


def test_exact_temporal_surface_binds_by_zero_based_relation_index() -> None:
    surface = "from  1 September 2026\nuntil 31 March 2027"
    result = _enrich(
        _payload(bindings=[{"relation_index": 0, "surface": surface}]),
        f"Policy applies {surface}.",
    )
    assert result["relations"][0]["qualifiers"]["temporality"] == surface


def test_temporal_surface_must_be_evidence_backed() -> None:
    with pytest.raises(ExtractionContractViolation) as captured:
        _enrich(
            _payload(bindings=[{"relation_index": 0, "surface": "tomorrow"}]),
            "Policy applies today.",
        )
    assert captured.value.code == "TEMPORAL_SURFACE_NOT_EVIDENCE_BACKED"


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda value: value["relations"][0].update(source_entity_index=2), "ENTITY_INDEX_OUT_OF_RANGE"),
        (lambda value: value.update(temporal_bindings=[{"relation_index": 1, "surface": "today"}]), "TEMPORAL_RELATION_INDEX_OUT_OF_RANGE"),
        (lambda value: value.update(temporal_bindings=[{"relation_index": 0, "surface": "today"}, {"relation_index": 0, "surface": "today"}]), "DUPLICATE_TEMPORAL_BINDING"),
    ],
)
def test_invalid_indices_and_duplicate_bindings_fail(
    mutate, code: str,
) -> None:
    payload = _payload()
    mutate(payload)
    with pytest.raises(ValidationError, match=code):
        SpanSemanticTemporalExtraction.model_validate(payload)


def test_status_contract_is_strict() -> None:
    invalid = _payload()
    invalid["status"] = "no_graph_fact"
    invalid["no_graph_fact_reason"] = "pure_context"
    with pytest.raises(ValidationError, match="no_graph_fact requires empty"):
        SpanSemanticTemporalExtraction.model_validate(invalid)
    valid = {
        "status": "no_graph_fact", "entities": [], "relations": [],
        "temporal_bindings": [], "no_graph_fact_reason": "pure_context",
    }
    assert SpanSemanticTemporalExtraction.model_validate(valid).status == "no_graph_fact"


def test_configured_semantic_prompt_is_selected_for_production() -> None:
    settings = load_project_settings(
        ROOT / "config",
        overrides={
            "extraction.contract_id": "evidence_span_fact_extraction",
            "extraction.prompt_id": "prompt_kimi_default",
        },
    )
    bundle = resolve_prompt(settings.extraction.prompt_id)
    assert bundle.contract is SpanSemanticTemporalExtraction
    assert settings.extraction_contracts[
        settings.extraction.contract_id
    ].schema_id == "fact_extraction_schema"
    production = load_project_settings(ROOT / "config")
    assert production.extraction.contract_id == "evidence_span_fact_extraction"
    assert production.extraction.prompt_id == "prompt_kimi_default"


@pytest.mark.parametrize(
    "case",
    json.loads((
        ROOT / "tests" / "fixtures" / "compact_repository_prompt_regressions.json"
    ).read_text(encoding="utf-8")),
    ids=lambda case: case["id"],
)
def test_generic_repository_outputs_validate_and_enrich(case: dict) -> None:
    payload = {
        "status": "facts_present",
        "entities": [
            {"canonical_name": name, "entity_type": "CONCEPT"}
            for name in case["entities"]
        ],
        "relations": [
            {
                "source_entity_index": source,
                "target_entity_index": target,
                "relation_family": family,
                "raw_relation": raw_relation,
                "condition": None, "modality": "ASSERTED",
                "quantity": None, "version": None, "negated": False,
            }
            for source, raw_relation, target, family in case["relations"]
        ],
        "temporal_bindings": [
            {"relation_index": relation_index, "surface": surface}
            for relation_index, surface in case["bindings"]
        ],
        "no_graph_fact_reason": None,
    }
    enriched = validate_and_enrich_semantic_temporal(
        SpanSemanticTemporalExtraction.model_validate(payload),
        unit_id="U001_S001", span_id="S001", evidence_text=case["source"],
    )
    assert [item["raw_relation"] for item in enriched["relations"]] == [
        item[1] for item in case["relations"]
    ]
    assert all(item["relation_family"] for item in enriched["relations"])
    assert [item["qualifiers"]["temporality"] for item in enriched["relations"]] == [
        next((surface for index, surface in case["bindings"] if index == cursor), None)
        for cursor in range(len(case["relations"]))
    ]
