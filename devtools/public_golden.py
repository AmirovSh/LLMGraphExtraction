"""Immutable public-sample semantic comparison and offline characterization gate."""
from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from config.settings import load_project_settings
from prompts.registry import resolve_prompt
from runtime.artifact_store import sha256_json
from runtime.final_graph_contract import validate_final_graph
from runtime.json_neo4j_parity import (
    EXPECTED_NODE_LABEL,
    EXPECTED_RELATIONSHIP_TYPE,
    build_parity_diff,
    neo4j_relationship_properties,
)
from runtime.production_runner import run

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tests" / "golden" / "public_sample_contract.json"


class OfflineResponse:
    status_code = 200
    content = b"offline structured response"
    text = ""

    def __init__(self, arguments: dict[str, Any]):
        self._body = {
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "submit_dynamic_inventory",
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        }
                    }]
                },
            }],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }

    def json(self) -> dict[str, Any]:
        return self._body

    def raise_for_status(self) -> None:
        return None


@contextmanager
def _offline_environment() -> Iterator[None]:
    values = {
        "SEMANTIC_GRAPH_LLM_BASE_URL": "https://offline.invalid",
        "SEMANTIC_GRAPH_EMBEDDING_BASE_URL": "https://offline.invalid",
        "OPENAI_API_KEY": "offline-placeholder",
        "NEO4J_URI": "bolt://offline.invalid:7687",
        "NEO4J_USERNAME": "offline",
        "NEO4J_PASSWORD": "offline-placeholder",
        "NEO4J_DATABASE": "neo4j",
    }
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _offline_resolution(entities, relations, **kwargs):
    del relations
    canonical_ids: dict[str, str] = {}
    mapping: dict[str, str] = {}
    for entity in entities:
        key = " ".join(entity["name"].split()).casefold()
        canonical_ids.setdefault(key, f"E{len(canonical_ids) + 1:03d}")
        mapping[entity["local_id"]] = canonical_ids[key]
    pair_count = len(entities) * (len(entities) - 1) // 2
    kwargs["write_artifact"]("entity_embedding_inputs.json", {
        "inputs": [entity["name"] for entity in entities],
        "input_count": len(entities),
        "mode": "offline_characterization_only",
    })
    return mapping, {
        "embedder_calls": 1,
        "usage": {"prompt_tokens": len(entities)},
        "retrieval_method": "offline_exact_name_characterization",
        "top_k": None,
        "candidate_threshold": .76,
        "merge_threshold": .90,
        "candidate_pairs": [],
        "merged_pairs": [],
        "keep_separate_pairs": [],
        "unresolved_merge_candidates": [],
        "all_pair_decision_counts": {},
        "vector_dimension": 1024,
        "embedding_input_count": len(entities),
        "unique_pair_count": pair_count,
    }


def _offline_neo4j(graph_id, nodes, edges, settings, ownership):
    projection = settings.graph_projection.type_names
    node_properties = {
        node["entity_id"]: {
            **node, "graph_id": graph_id, "display_name": node["canonical_name"],
            **ownership,
        }
        for node in nodes
    }
    edge_properties = {
        edge["edge_id"]: neo4j_relationship_properties(edge, graph_id, ownership)
        for edge in edges
    }
    parity = build_parity_diff(
        graph_id=graph_id,
        json_nodes=nodes,
        json_edges=edges,
        neo4j_node_ids=set(node_properties),
        neo4j_edges={
            edge["edge_id"]: (edge["source_entity_id"], edge["target_entity_id"])
            for edge in edges
        },
        edges_without_graph_id=[],
        duplicate_edges=[],
        import_errors=[], ownership=ownership,
        neo4j_node_properties=node_properties,
        neo4j_edge_properties=edge_properties,
        active_project_labels={str(projection.entity_label)},
        active_project_relationship_types={str(projection.relation_type)},
        project_indexes=[
            {"name": projection.entity_index, "labelsOrTypes": [str(projection.entity_label)]},
            {"name": projection.relation_index, "labelsOrTypes": [str(projection.relation_type)]},
        ],
        projection=projection,
    )
    return {"graph_id": graph_id, "nodes": len(nodes), "edges": len(edges)}, parity


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_public_contract(
    *, graph: dict[str, Any], budget: dict[str, Any], parity: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    validated = validate_final_graph(graph).model_dump(mode="json")
    errors: list[str] = []
    observed_fixture = (validated["source"].get("repository_relative_path") or "").replace("\\", "/")
    expected_fixture = manifest["fixture"].replace("\\", "/")
    if observed_fixture != expected_fixture:
        errors.append("source fixture differs from public golden contract")
    if validated["prompt_id"] != manifest["prompt_id"]:
        errors.append("prompt_id differs from public golden contract")
    if validated["prompt_hash"] != manifest["prompt_hash"]:
        errors.append("prompt_hash differs from public golden contract")
    name_to_ids: dict[str, set[str]] = {}
    for node in validated["nodes"]:
        name_to_ids.setdefault(node["canonical_name"], set()).add(node["entity_id"])
        for alias in node["aliases"]:
            name_to_ids.setdefault(alias, set()).add(node["entity_id"])
    for name in manifest["required_entities"]:
        if name not in name_to_ids:
            errors.append(f"missing required entity: {name}")
    for merge in manifest["required_alias_merges"]:
        canonical = merge["canonical_name"]
        alias = merge["alias"]
        if not (
            canonical in name_to_ids
            and alias in name_to_ids
            and name_to_ids[canonical] & name_to_ids[alias]
        ):
            errors.append(f"required alias merge missing: {canonical} / {alias}")

    relations: list[tuple[str, str, str, bool, dict[str, Any]]] = []
    node_names = {
        node["entity_id"]: node["canonical_name"] for node in validated["nodes"]
    }
    for edge in validated["edges"]:
        relations.append((
            node_names[edge["source_entity_id"]],
            edge["raw_relation"],
            node_names[edge["target_entity_id"]],
            edge["qualifiers"]["negated"],
            edge["qualifiers"],
        ))
        if not edge["evidence_span_ids"] or edge["provenance"]["source"] != "fact_extraction":
            errors.append(f"{edge['edge_id']}: missing extraction evidence lineage")
    relation_keys = {(source, relation, target, negated) for source, relation, target, negated, _ in relations}
    for expected in manifest["required_relations"]:
        key = (
            expected["source"], expected["raw_relation"], expected["target"],
            expected.get("negated", False),
        )
        if key not in relation_keys:
            errors.append(f"missing required relation: {key}")
    for forbidden in manifest["forbidden_relations"]:
        if any(
            source == forbidden["source"]
            and relation == forbidden["raw_relation"]
            and target == forbidden["target"]
            for source, relation, target, _, _ in relations
        ):
            errors.append(f"forbidden relation present: {forbidden}")
    for left, right in manifest["required_separate_entities"]:
        if left in name_to_ids and right in name_to_ids and name_to_ids[left] & name_to_ids[right]:
            errors.append(f"entities must remain separate: {left} / {right}")
    for case in manifest["required_qualifier_cases"]:
        matching = [
            qualifiers for source, relation, _, _, qualifiers in relations
            if source == case["source"] and relation == case["raw_relation"]
        ]
        if not matching or matching[0].get(case["field"]) != case["value"]:
            errors.append(f"required qualifier case failed: {case}")

    totals = budget["totals"]
    for field, expected in manifest["call_budget"].items():
        actual = len(validated["source"]["units"]) if field == "unit_count" else totals.get(field)
        if actual != expected:
            errors.append(f"call budget {field}: expected {expected}, got {actual}")
    if parity.get("status") != manifest["projection_contract"]["parity_status"]:
        errors.append("projection parity status differs from golden contract")
    projection = manifest["projection_contract"]
    schema = parity.get("schema_parity") or {}
    if projection["node_label"] not in schema.get("active_project_labels", []):
        errors.append("required Neo4j node label is absent")
    if projection["relationship_type"] not in schema.get(
        "active_project_relationship_types", []
    ):
        errors.append("required Neo4j relationship type is absent")
    if schema.get("missing_project_indexes"):
        errors.append("required Neo4j project index is absent")
    if any(not edge["raw_relation"].strip() for edge in validated["edges"]):
        errors.append("raw_relation coverage is incomplete")

    graph_hash = sha256_json(validated)
    required_relation_failures = sum(
        error.startswith("missing required relation:") for error in errors
    )
    required_qualifier_failures = sum(
        error.startswith("required qualifier case failed:") for error in errors
    )
    forbidden_failures = sum(
        error.startswith("forbidden relation present:") for error in errors
    )
    required_total = (
        len(manifest["required_relations"])
        + len(manifest["required_qualifier_cases"])
    )
    forbidden_total = len(manifest["forbidden_relations"])
    required_passed = (
        required_total - required_relation_failures - required_qualifier_failures
    )
    forbidden_passed = forbidden_total - forbidden_failures
    relation_families: dict[str, int] = {}
    for edge in validated["edges"]:
        family = edge["relation_family"]
        relation_families[family] = relation_families.get(family, 0) + 1
    return {
        "status": "passed" if not errors else "failed",
        "failed_assertions": errors,
        "assertion_counts": {
            "required_assertions_passed": required_passed,
            "required_assertions_total": required_total,
            "forbidden_assertions_passed": forbidden_passed,
            "forbidden_assertions_total": forbidden_total,
            "passed": required_passed + forbidden_passed,
            "total": required_total + forbidden_total,
        },
        "observed_metrics": {
            "graph_hash": graph_hash,
            "nodes": len(validated["nodes"]),
            "edges": len(validated["edges"]),
            "unit_count": len(validated["source"]["units"]),
            "primary_http_attempts": totals.get("primary_http_attempts"),
            "primary_successful_calls": totals.get("primary_successful_calls"),
            "automatic_retries": totals.get("automatic_retries"),
            "transport_successes": totals.get("transport_successes"),
            "transport_failures": totals.get("transport_failures"),
            "provider_response_rejections": totals.get("provider_response_rejections"),
            "tool_contract_failures": totals.get("tool_contract_failures"),
            "schema_failures": totals.get("schema_failures"),
            "evidence_failures": totals.get("evidence_failures"),
            "accepted_extractions": totals.get("accepted_extractions"),
            "post_extraction_llm_calls": totals.get("post_extraction_llm_calls"),
            "embedding_calls": totals.get("embedding_calls"),
            "semantic_assertions_passed": not errors,
            "projection_parity_status": parity.get("status"),
            "relation_family_distribution": dict(sorted(relation_families.items())),
        },
        "contract_version": manifest["contract_version"],
    }


def run_offline_public_golden(
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    structured_path = ROOT / manifest["structured_fixture"]
    structured = json.loads(structured_path.read_text(encoding="utf-8"))
    expected_units = structured["units"]
    expected_by_unit = {
        item["unit_id"]: item["arguments"] for item in expected_units
    }
    consumed: set[str] = set()
    lock = threading.Lock()

    def post(*args, **kwargs):
        del args
        payload = kwargs["json"]
        user_prompt = payload["messages"][-1]["content"]
        matches = [
            unit_id for unit_id in expected_by_unit
            if (
                f'"span_id":"{unit_id.rsplit("_", 1)[-1]}"'
                in user_prompt
            )
        ]
        if len(matches) != 1:
            raise AssertionError("offline golden received an unknown span unit")
        unit_id = matches[0]
        with lock:
            if unit_id in consumed:
                raise AssertionError("offline golden span unit was reissued")
            consumed.add(unit_id)
        return OfflineResponse(expected_by_unit[unit_id])

    with tempfile.TemporaryDirectory(prefix="public_golden_offline_") as temporary:
        output = Path(temporary) / "public_golden_offline"
        with _offline_environment():
            run(
                ROOT / manifest["fixture"],
                output,
                ROOT / "config",
                run_id="offline_public_golden",
                llm_post=post,
                embedding_resolver=_offline_resolution,
                neo4j_importer=_offline_neo4j,
            )
        graph = json.loads((output / "fact_graph.json").read_text(encoding="utf-8"))
        budget = json.loads((output / "model_call_budget.json").read_text(encoding="utf-8"))
        parity = json.loads((output / "json_neo4j_edge_diff.json").read_text(encoding="utf-8"))
        result = compare_public_contract(
            graph=graph, budget=budget, parity=parity, manifest=manifest,
        )
        result.update({
            "mode": "offline",
            "live_model_acceptance": False,
            "clean_namespace": True,
            "resume_used": False,
            "structured_units_consumed": len(consumed),
        })
        return result


def check_artifact_run(
    run_dir: Path, manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    graph = json.loads((run_dir / "fact_graph.json").read_text(encoding="utf-8"))
    budget = json.loads((run_dir / "model_call_budget.json").read_text(encoding="utf-8"))
    parity = json.loads((run_dir / "json_neo4j_edge_diff.json").read_text(encoding="utf-8"))
    result = compare_public_contract(
        graph=graph, budget=budget, parity=parity, manifest=manifest,
    )
    result.update({"mode": "artifact", "run_dir": str(run_dir)})
    return result


def prompt_contract_metadata(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, str]:
    manifest = load_manifest(manifest_path)
    settings = load_project_settings(ROOT / "config")
    bundle = resolve_prompt(settings.extraction.prompt_id)
    return {
        "expected_prompt_id": manifest["prompt_id"],
        "actual_prompt_id": bundle.prompt_id,
        "actual_prompt_hash": bundle.content_hash,
    }
