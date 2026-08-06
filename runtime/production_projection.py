"""Direct, non-semantic projection of validated fact-extraction output."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from prompts.contracts import FactExtraction

LOCAL_ID_PATTERN = re.compile(r"\bM[0-9]{3}\b")


def _stable_id(prefix: str, payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


def _signature(relation: Mapping[str, Any]) -> tuple[object, ...]:
    qualifiers = relation["qualifiers"]
    return (
        relation["source_entity_id"], relation["raw_relation"], relation["target_entity_id"],
        tuple(sorted(relation["evidence_span_ids"])), json.dumps(qualifiers, ensure_ascii=False, sort_keys=True),
        "forward", qualifiers["negated"], qualifiers["modality"],
    )


def rebind_qualifier_entity_references(qualifiers: Mapping[str, Any], resolved_entity_ids: Mapping[str, str]) -> dict[str, Any]:
    """Rebind exact local-ID qualifier references and reject every unresolved local ID."""
    rebound = deepcopy(dict(qualifiers))
    for key,value in rebound.items():
        if isinstance(value,str) and value in resolved_entity_ids:
            rebound[key] = resolved_entity_ids[value]
        elif isinstance(value,str) and LOCAL_ID_PATTERN.search(value):
            raise ValueError(f"unresolved local entity ID in qualifier {key}: {value}")
    return rebound


def project_extraction_unit(
    *, unit_id: str, extraction: FactExtraction, resolved_entity_ids: Mapping[str, str],
    valid_span_ids: set[str], prompt_identity: Mapping[str, object],
) -> dict[str, object]:
    """Serialize direct output after technical validation and established resolution.

    This function intentionally cannot rewrite predicates/endpoints or add facts.
    """
    for entity in extraction.entities:
        if entity.local_id not in resolved_entity_ids:
            raise ValueError(f"missing resolved entity id for {entity.local_id}")
        if not set(entity.evidence_span_ids) <= valid_span_ids:
            raise ValueError(f"unknown entity evidence span in {entity.local_id}")
    edges: list[dict[str, Any]] = []
    seen: set[tuple[object, ...]] = set()
    for relation in extraction.relations:
        if not set(relation.evidence_span_ids) <= valid_span_ids:
            raise ValueError("unknown relation evidence span")
        edge = {
            "source_entity_id": resolved_entity_ids[relation.source_local_id],
            "target_entity_id": resolved_entity_ids[relation.target_local_id],
            "raw_relation": relation.raw_relation,
            "relation_family": relation.relation_family,
            "relation_description": relation.relation_description,
            "evidence_span_ids": list(relation.evidence_span_ids),
            "support_level": relation.support_level,
            "qualifiers": rebind_qualifier_entity_references(relation.qualifiers.model_dump(mode="json"),resolved_entity_ids),
            "unit_id": unit_id,
            "provenance": {
                "prompt_id": prompt_identity["prompt_id"],
                "source": "fact_extraction",
            },
        }
        signature = _signature(edge)
        if signature in seen:
            continue
        seen.add(signature)
        edge["edge_id"] = _stable_id("R", signature)
        edges.append(edge)
    nodes = [
        {
            "entity_id": resolved_entity_ids[item.local_id], "name": item.name, "type": item.type,
            "evidence_span_ids": list(item.evidence_span_ids), "unit_id": unit_id,
            "provenance": {
                "prompt_id": prompt_identity["prompt_id"],
                "source": "fact_extraction",
            },
        }
        for item in extraction.entities
    ]
    return {"unit_id": unit_id, "nodes": deepcopy(nodes), "edges": deepcopy(edges)}
