"""Typed resolver for the supported production extraction prompt."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import BaseModel

from prompts.contracts import (
    FactExtraction,
    SpanSemanticTemporalExtraction,
    build_tool_schema,
)


@dataclass(frozen=True)
class PromptBundle:
    prompt_id: str
    template: str
    ontology: dict[str, Any]
    contract: type[BaseModel]
    schema: dict[str, Any]
    schema_builder: Callable[..., dict[str, Any]]
    graph_adapter: Callable[[BaseModel], FactExtraction]
    content_hash: str


def _adapt_to_graph(value: BaseModel) -> FactExtraction:
    payload = value.model_dump(mode="json")
    bindings = {
        item["relation_index"]: item["surface"]
        for item in payload["temporal_bindings"]
    }
    entities = [
        {
            "local_id": f"M{index + 1:03d}",
            "name": item["canonical_name"],
            "type": item["entity_type"],
            "evidence_span_ids": ["S001"],
        }
        for index, item in enumerate(payload["entities"])
    ]
    relations = [
        {
            "source_local_id": entities[item["source_entity_index"]]["local_id"],
            "raw_relation": item["raw_relation"],
            "relation_family": item["relation_family"],
            "relation_description": item["raw_relation"],
            "target_local_id": entities[item["target_entity_index"]]["local_id"],
            "evidence_span_ids": ["S001"],
            "support_level": "EXPLICIT",
            "qualifiers": {
                "temporality": bindings.get(index),
                "condition": item["condition"],
                "modality": item["modality"] or "ASSERTED",
                "negated": item["negated"],
                "quantity": item["quantity"],
                "version": item["version"],
            },
        }
        for index, item in enumerate(payload["relations"])
    ]
    return FactExtraction.model_validate({"entities": entities, "relations": relations})


def _production_bundle() -> PromptBundle:
    root = Path(__file__).parent / "fact_extraction"
    prompt_path = root / "prompt_kimi_default.md"
    template = prompt_path.read_text(encoding="utf-8")
    ontology = yaml.safe_load((root / "ontology.yaml").read_text(encoding="utf-8"))
    schema = build_tool_schema(
        max_entities=60,
        max_relations=40,
        max_evidence_spans=10,
        max_relation_description_characters=300,
    )
    content = json.dumps(
        {"template": template, "ontology": ontology, "schema": schema},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return PromptBundle(
        prompt_id=prompt_path.stem,
        template=template,
        ontology=ontology,
        contract=SpanSemanticTemporalExtraction,
        schema=schema,
        schema_builder=build_tool_schema,
        graph_adapter=_adapt_to_graph,
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


_PRODUCTION_BUNDLE = _production_bundle()


def resolve_prompt(prompt_id: str) -> PromptBundle:
    """Resolve a configured semantic prompt identifier before transport."""
    if prompt_id != _PRODUCTION_BUNDLE.prompt_id:
        raise ValueError(
            f"unknown prompt_id {prompt_id!r}; "
            f"available: {_PRODUCTION_BUNDLE.prompt_id}"
        )
    return _PRODUCTION_BUNDLE
