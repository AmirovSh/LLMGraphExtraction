"""Stable extraction models shared by the final prompt and graph pipeline."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoGraphFactReason(str, Enum):
    HEADING_OR_FRAGMENT = "heading_or_fragment"
    PURE_CONTEXT = "pure_context"
    DUPLICATE_STATEMENT = "duplicate_statement"
    NO_BINARY_RELATION = "no_binary_relation"


class Qualifiers(BaseModel):
    temporality: str | None = None
    condition: str | None = None
    modality: Literal["ASSERTED", "REQUIRED", "POSSIBLE", "PLANNED", "HISTORICAL"]
    negated: bool = Field(description=(
        "True only when the relation proposition itself is explicitly denied. "
        "Positive blocks, prevents, rejects, and disables propositions are false."
    ))
    quantity: str | None = None
    version: str | None = None


class LocalEntity(BaseModel):
    local_id: str = Field(pattern=r"^M[0-9]{3}$")
    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    evidence_span_ids: list[str] = Field(min_length=1, max_length=10)


class LocalRelation(BaseModel):
    source_local_id: str = Field(pattern=r"^M[0-9]{3}$")
    raw_relation: str = Field(min_length=1)
    relation_family: str = Field(min_length=1)
    relation_description: str = Field(min_length=1, max_length=300)
    target_local_id: str = Field(pattern=r"^M[0-9]{3}$")
    evidence_span_ids: list[str] = Field(min_length=1, max_length=10)
    support_level: Literal["EXPLICIT", "LOCALLY_ENTAILED", "PLAUSIBLE"]
    qualifiers: Qualifiers


class FactExtraction(BaseModel):
    entities: list[LocalEntity] = Field(max_length=60)
    relations: list[LocalRelation] = Field(max_length=40)

    @model_validator(mode="after")
    def validate_local_references(self) -> "FactExtraction":
        ids = [entity.local_id for entity in self.entities]
        if len(ids) != len(set(ids)):
            raise ValueError("local entity IDs must be unique")
        known = set(ids)
        if any(
            relation.source_local_id not in known
            or relation.target_local_id not in known
            for relation in self.relations
        ):
            raise ValueError("relations must reference declared local entity IDs")
        return self


class RelationFamily(str, Enum):
    COMPOSITION = "COMPOSITION"
    DATA_FLOW = "DATA_FLOW"
    PROCESSING_SEQUENCE = "PROCESSING_SEQUENCE"
    DEPENDENCY = "DEPENDENCY"
    VALIDATION_QUALITY = "VALIDATION_QUALITY"
    RESPONSIBILITY_OWNERSHIP = "RESPONSIBILITY_OWNERSHIP"
    LIFECYCLE_VERSION = "LIFECYCLE_VERSION"
    CONDITION_REQUIREMENT = "CONDITION_REQUIREMENT"
    CAUSAL_RESULT = "CAUSAL_RESULT"
    ASSERTION_CONTEXT = "ASSERTION_CONTEXT"


class SemanticEntity(StrictContractModel):
    canonical_name: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)


class SemanticRelation(StrictContractModel):
    source_entity_index: int = Field(ge=0)
    target_entity_index: int = Field(ge=0)
    relation_family: RelationFamily
    raw_relation: str = Field(min_length=1)
    condition: str | None
    modality: Literal["ASSERTED", "REQUIRED", "POSSIBLE", "PLANNED", "HISTORICAL"] | None
    quantity: str | None
    version: str | None
    negated: bool


class SemanticTemporalBinding(StrictContractModel):
    relation_index: int = Field(ge=0)
    surface: str = Field(min_length=1)


class SpanSemanticTemporalExtraction(StrictContractModel):
    status: Literal["facts_present", "no_graph_fact"]
    entities: list[SemanticEntity] = Field(max_length=60)
    relations: list[SemanticRelation] = Field(max_length=40)
    temporal_bindings: list[SemanticTemporalBinding]
    no_graph_fact_reason: NoGraphFactReason | None

    @model_validator(mode="after")
    def validate_status_and_indices(self) -> "SpanSemanticTemporalExtraction":
        if self.status == "facts_present":
            if not self.entities or not self.relations:
                raise ValueError("facts_present requires non-empty entities and relations")
            if self.no_graph_fact_reason is not None:
                raise ValueError("facts_present forbids no_graph_fact_reason")
        else:
            if self.entities or self.relations or self.temporal_bindings:
                raise ValueError("no_graph_fact requires empty graph arrays")
            if self.no_graph_fact_reason is None:
                raise ValueError("no_graph_fact requires a typed reason")
        entity_count = len(self.entities)
        if any(
            relation.source_entity_index >= entity_count
            or relation.target_entity_index >= entity_count
            for relation in self.relations
        ):
            raise ValueError("ENTITY_INDEX_OUT_OF_RANGE")
        relation_count = len(self.relations)
        if any(item.relation_index >= relation_count for item in self.temporal_bindings):
            raise ValueError("TEMPORAL_RELATION_INDEX_OUT_OF_RANGE")
        indices = [item.relation_index for item in self.temporal_bindings]
        if len(indices) != len(set(indices)):
            raise ValueError("DUPLICATE_TEMPORAL_BINDING")
        return self


def build_tool_schema(
    *, max_entities: int, max_relations: int, max_evidence_spans: int,
    max_relation_description_characters: int,
) -> dict[str, Any]:
    del max_evidence_spans, max_relation_description_characters
    schema = SpanSemanticTemporalExtraction.model_json_schema()
    schema["properties"]["entities"]["maxItems"] = max_entities
    schema["properties"]["relations"]["maxItems"] = max_relations
    return schema
