"""Typed, non-semantic contract for the authoritative final fact graph."""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LOCAL_ID_PATTERN = re.compile(r"\bM[0-9]{3}\b")


class StrictContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FinalGraphEvidence(StrictContractModel):
    evidence_span_ids: list[str] = Field(min_length=1)

    @field_validator("evidence_span_ids")
    @classmethod
    def validate_evidence_span_ids(cls, value: list[str]) -> list[str]:
        if any(not item for item in value):
            raise ValueError("evidence span IDs must be non-empty strings")
        return value


class FinalGraphProvenance(StrictContractModel):
    prompt_id: str = Field(min_length=1)
    source: Literal["fact_extraction"]


class FinalGraphQualifiers(StrictContractModel):
    temporality: str | None = None
    condition: str | None = None
    modality: Literal["ASSERTED", "REQUIRED", "POSSIBLE", "PLANNED", "HISTORICAL"]
    negated: bool
    quantity: str | None = None
    version: str | None = None

    @model_validator(mode="after")
    def reject_unresolved_local_ids(self) -> "FinalGraphQualifiers":
        for field_name, value in self:
            if isinstance(value, str) and LOCAL_ID_PATTERN.search(value):
                raise ValueError(f"unresolved local entity ID in qualifier {field_name}: {value}")
        return self


class FinalGraphNode(FinalGraphEvidence):
    entity_id: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    primary_type: str = Field(min_length=1)
    aliases: list[str]
    local_entity_ids: list[str] = Field(min_length=1)

    @field_validator("entity_id")
    @classmethod
    def reject_local_entity_id(cls, value: str) -> str:
        if LOCAL_ID_PATTERN.search(value):
            raise ValueError("global entity_id contains an unresolved local entity ID")
        return value


class FinalGraphEdge(FinalGraphEvidence):
    source_entity_id: str = Field(min_length=1)
    target_entity_id: str = Field(min_length=1)
    raw_relation: str = Field(min_length=1)
    relation_family: str = Field(min_length=1)
    relation_description: str = Field(min_length=1)
    support_level: Literal["EXPLICIT", "LOCALLY_ENTAILED", "PLAUSIBLE"]
    qualifiers: FinalGraphQualifiers
    unit_id: str = Field(min_length=1)
    provenance: FinalGraphProvenance
    edge_id: str = Field(min_length=1)

    @field_validator("raw_relation", "relation_family")
    @classmethod
    def reject_blank_contract_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string after whitespace validation")
        return value

    @field_validator("source_entity_id", "target_entity_id")
    @classmethod
    def reject_local_endpoint_id(cls, value: str) -> str:
        if LOCAL_ID_PATTERN.search(value):
            raise ValueError("global endpoint contains an unresolved local entity ID")
        return value


class FinalGraphSourceUnit(StrictContractModel):
    unit_id: str = Field(min_length=1)
    span_ids: list[str] = Field(min_length=1)


class FinalGraphSource(StrictContractModel):
    source_name: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_size_bytes: int = Field(ge=0)
    source_location_policy: Literal["repository_relative", "external"]
    repository_relative_path: str | None = None
    units: list[FinalGraphSourceUnit] = Field(min_length=1)

    @model_validator(mode="after")
    def relative_path_matches_policy(self) -> "FinalGraphSource":
        if self.source_location_policy == "repository_relative":
            if not self.repository_relative_path:
                raise ValueError("repository-relative source requires a relative path")
        elif self.repository_relative_path is not None:
            raise ValueError("external source must not persist a local path")
        return self


class FinalGraph(StrictContractModel):
    graph_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    prompt_hash: str = Field(min_length=1)
    manifest_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    nodes: list[FinalGraphNode]
    edges: list[FinalGraphEdge]
    source: FinalGraphSource

    @model_validator(mode="after")
    def validate_graph_integrity(self) -> "FinalGraph":
        node_ids = [node.entity_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("duplicate node ID")
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("duplicate edge ID")
        known_nodes = set(node_ids)
        unit_ids = {unit.unit_id for unit in self.source.units}
        valid_span_ids = {
            span_id for unit in self.source.units for span_id in unit.span_ids
        }
        for node in self.nodes:
            if not set(node.evidence_span_ids) <= valid_span_ids:
                raise ValueError(f"{node.entity_id}: evidence span is absent from graph source")
        for edge in self.edges:
            if edge.source_entity_id not in known_nodes:
                raise ValueError(f"{edge.edge_id}: source endpoint is absent from nodes")
            if edge.target_entity_id not in known_nodes:
                raise ValueError(f"{edge.edge_id}: target endpoint is absent from nodes")
            if edge.unit_id not in unit_ids:
                raise ValueError(f"{edge.edge_id}: unit_id is absent from graph source")
            if not set(edge.evidence_span_ids) <= valid_span_ids:
                raise ValueError(f"{edge.edge_id}: evidence span is absent from graph source")
            if edge.provenance.prompt_id != self.prompt_id:
                raise ValueError(f"{edge.edge_id}: prompt_id differs from graph contract")
        return self


def validate_final_graph(candidate: dict[str, Any]) -> FinalGraph:
    """Validate without normalizing or repairing any persisted value."""
    return FinalGraph.model_validate(candidate)
