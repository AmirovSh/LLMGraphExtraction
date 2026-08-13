"""Pre-resolution entity profiles for the isolated Phase-3R experiment."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

Representation = Literal["R1", "R2", "R3", "R4"]
FORBIDDEN_FIELDS = frozenset({"global_entity_id", "final_graph", "resolution_decision", "merge_result",
    "split_result", "candidate_label", "expected", "resolved_neighbours", "neo4j", "candidate_scores"})


class LocalIncident(BaseModel):
    model_config = ConfigDict(extra="forbid")
    direction: Literal["outgoing", "incoming"]
    relation: str = Field(min_length=1)
    neighbour_name: str = Field(min_length=1)
    source_order: int = Field(ge=0)


class LocalEntityContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    local_id: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    incidents: list[LocalIncident] = Field(default_factory=list)
    source_kind: Literal["REAL_ARTIFACT", "CONTROLLED_SYNTHETIC"]
    identity_cluster_id: str = Field(min_length=1)


def _clean(value: str) -> str:
    return " ".join(value.split())


def render_profile(entity: LocalEntityContext, representation: Representation, *, evidence_limit: int = 5,
                   incident_limit: int = 8, snippet_limit: int = 160, total_limit: int = 800) -> dict[str, Any]:
    if representation == "R1":
        lines = [_clean(entity.canonical_name)]
    else:
        aliases = [_clean(v) for v in entity.aliases if _clean(v)]
        lines = [f"canonical_name: {_clean(entity.canonical_name)}", f"type: {_clean(entity.entity_type)}", "aliases:"]
        lines.extend(f"- {v}" for v in aliases or ["none"])
        if representation in {"R3", "R4"}:
            snippets = [_clean(v)[:snippet_limit] for v in entity.evidence[:evidence_limit] if _clean(v)]
            lines.append("evidence:"); lines.extend(f"- {v}" for v in snippets or ["none"])
        if representation == "R4":
            lines.append("observed_local_relations:")
            incidents = sorted(entity.incidents, key=lambda x: x.source_order)[:incident_limit]
            for item in incidents:
                arrow = "->" if item.direction == "outgoing" else "<-"
                lines.append(f"- {item.relation} {arrow} {_clean(item.neighbour_name)}")
            if not incidents: lines.append("- none")
    original_text = "\n".join(lines); text = original_text[:total_limit]
    return {"local_id": entity.local_id, "representation": representation, "text": text,
            "original_characters": len(original_text), "retained_characters": len(text),
            "truncated": len(original_text) > total_limit, "approx_tokens": max(1, (len(text) + 3) // 4)}


def eligibility(entity: LocalEntityContext) -> dict[str, bool]:
    return {"R1_ELIGIBLE": bool(entity.canonical_name), "R2_ELIGIBLE": bool(entity.canonical_name and entity.entity_type),
            "R3_ELIGIBLE": bool(entity.canonical_name and entity.entity_type and entity.evidence),
            "R4_ELIGIBLE": bool(entity.canonical_name and entity.entity_type and entity.evidence and entity.incidents)}


def assert_pre_resolution_payload(payload: Mapping[str, Any]) -> None:
    leaked = FORBIDDEN_FIELDS & {str(key).casefold() for key in payload}
    if leaked: raise ValueError(f"post-resolution fields are forbidden: {sorted(leaked)}")


def serializer_contract() -> dict[str, Any]:
    contract = {"version": "phase3r-profile-v1", "historical_compatibility": "reconstructed-compatible",
        "field_order": ["canonical_name", "type", "aliases", "evidence", "observed_local_relations"],
        "evidence_order": "source order", "evidence_limit": 5, "snippet_character_limit": 160,
        "incident_order": "source order", "incident_limit": 8, "total_character_limit": 800,
        "pair_independent": True, "pre_resolution_only": True}
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return contract | {"sha256": hashlib.sha256(encoded).hexdigest()}
