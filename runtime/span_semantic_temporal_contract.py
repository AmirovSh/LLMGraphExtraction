"""Validate compact semantic output and deterministically enrich it."""
from __future__ import annotations

import unicodedata
from typing import Any

from prompts.contracts import SpanSemanticTemporalExtraction
from runtime.span_extraction import ExtractionContractViolation


def _neutral_surface(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def validate_and_enrich_semantic_temporal(
    value: SpanSemanticTemporalExtraction, *, unit_id: str, span_id: str,
    evidence_text: str,
) -> dict[str, Any]:
    evidence = _neutral_surface(evidence_text)
    seen: set[int] = set()
    for binding in value.temporal_bindings:
        if binding.relation_index >= len(value.relations):
            raise ExtractionContractViolation(
                "TEMPORAL_RELATION_INDEX_OUT_OF_RANGE", str(binding.relation_index)
            )
        if binding.relation_index in seen:
            raise ExtractionContractViolation(
                "DUPLICATE_TEMPORAL_BINDING", str(binding.relation_index)
            )
        if _neutral_surface(binding.surface) not in evidence:
            raise ExtractionContractViolation(
                "TEMPORAL_SURFACE_NOT_EVIDENCE_BACKED", str(binding.relation_index)
            )
        seen.add(binding.relation_index)

    prefix = f"{span_id}_"
    entities = [
        {
            "local_id": f"{prefix}M{index + 1:03d}",
            "name": item.canonical_name,
            "type": item.entity_type,
            "evidence_span_ids": [span_id],
        }
        for index, item in enumerate(value.entities)
    ]
    temporal = {
        item.relation_index: item.surface for item in value.temporal_bindings
    }
    relations = []
    for index, item in enumerate(value.relations):
        relations.append({
            "relation_local_id": f"{prefix}R{index + 1:03d}",
            "source_local_id": entities[item.source_entity_index]["local_id"],
            "raw_relation": item.raw_relation,
            "relation_family": item.relation_family.value,
            "relation_description": item.raw_relation,
            "target_local_id": entities[item.target_entity_index]["local_id"],
            "evidence_span_ids": [span_id],
            "support_level": "EXPLICIT",
            "qualifiers": {
                "temporality": temporal.get(index),
                "condition": item.condition,
                "modality": item.modality or "ASSERTED",
                "negated": item.negated,
                "quantity": item.quantity,
                "version": item.version,
            },
        })
    return {
        "unit_id": unit_id,
        "span_id": span_id,
        "status": value.status,
        "no_graph_fact_reason": (
            value.no_graph_fact_reason.value if value.no_graph_fact_reason else None
        ),
        "entities": entities,
        "relations": relations,
        "temporal_bindings": [
            {"relation_index": item.relation_index, "surface": item.surface}
            for item in value.temporal_bindings
        ],
    }
