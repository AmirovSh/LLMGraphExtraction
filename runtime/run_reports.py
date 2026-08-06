"""Pure production graph metrics and report assembly."""
from __future__ import annotations

from typing import Any


def component_count(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> int:
    adjacent = {item["entity_id"]:set() for item in nodes}
    for edge in edges:
        adjacent[edge["source_entity_id"]].add(edge["target_entity_id"]); adjacent[edge["target_entity_id"]].add(edge["source_entity_id"])
    seen: set[str] = set(); count = 0
    for node in adjacent:
        if node in seen: continue
        count += 1; stack = [node]
        while stack:
            current = stack.pop()
            if current not in seen: seen.add(current); stack.extend(adjacent[current]-seen)
    return count


def isolated_node_ids(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[str]:
    connected = {endpoint for edge in edges for endpoint in (edge["source_entity_id"],edge["target_entity_id"])}
    return sorted(node["entity_id"] for node in nodes if node["entity_id"] not in connected)


def deduplicate_exact_edges(edges: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unique: dict[str,dict[str,Any]] = {}; duplicates = []
    for edge in edges:
        if edge["edge_id"] in unique: duplicates.append(edge)
        else: unique[edge["edge_id"]] = edge
    return list(unique.values()), duplicates


def validation_report(*, choices: list[dict[str, Any]], transports: list[str], local_relations: list[dict[str, Any]], valid_spans: set[str], budget: dict[str, Any], parity: dict[str, Any]) -> dict[str, Any]:
    finish_reasons = [choice.get("finish_reason") for choice in choices]
    report = {"passed":all(reason in {"tool_calls","stop"} for reason in finish_reasons),"registered_prompt_active":True,"fact_layer_calls":0,
              "post_extraction_llm_calls":budget["totals"]["post_extraction_llm_calls"],
              "unknown_evidence_span_ids":sorted({span for item in local_relations for span in item["evidence_span_ids"]}-valid_spans),
              "finish_reasons":finish_reasons,"transports":transports,"llm_merge_verifier_calls":0,"gleaning_calls":0,"semantic_patch_calls":0,
              "json_neo4j_parity":parity.get("status") == "passed" if "status" in parity else not any(
                  parity[key] for key in ("json_only_nodes","neo4j_only_nodes","json_only_edges",
                                          "neo4j_only_edges","source_target_mismatches","import_errors"))}
    report["passed"] = report["passed"] and not report["unknown_evidence_span_ids"] and report["post_extraction_llm_calls"] == 0 and report["json_neo4j_parity"]
    return report


def run_report(*, run_id: str, graph_id: str, source: str, units: list[str], choices: list[dict[str, Any]],
               budget: dict[str, Any], resolution: dict[str, Any], local_entities: list[dict[str, Any]],
               local_relations: list[dict[str, Any]], entities: list[dict[str, Any]], edges: list[dict[str, Any]],
               neo4j: dict[str, Any], validation: dict[str, Any], transports: list[str]) -> dict[str, Any]:
    totals=budget["totals"]
    return {"run_id":run_id,"graph_id":graph_id,"source_words":len(source.split()),"units":len(units),
            "primary_successful_calls":totals["primary_successful_calls"],"primary_http_attempts":totals["primary_http_attempts"],
            "primary_failed_attempts":totals["primary_failed_attempts"],"primary_timeout_attempts":totals["primary_timeout_attempts"],
            "automatic_retries":totals["automatic_retries"],
            "transport_successes":totals["transport_successes"],"transport_failures":totals["transport_failures"],
            "provider_response_rejections":totals["provider_response_rejections"],"tool_contract_failures":totals["tool_contract_failures"],
            "schema_failures":totals["schema_failures"],"evidence_failures":totals["evidence_failures"],"accepted_extractions":totals["accepted_extractions"],
            "post_extraction_llm_calls":totals["post_extraction_llm_calls"],
            "prompt_tokens":budget["totals"]["prompt_tokens"],"completion_tokens":budget["totals"]["completion_tokens"],"total_tokens":budget["totals"]["total_tokens"],
            "embedding_usage":resolution.get("usage") or {},"finish_reasons":[choice.get("finish_reason") for choice in choices],"transports":transports,
            "local_entities":len(local_entities),"local_relations":len(local_relations),"global_entities":len(entities),"merged_entities":resolution["merged_pairs"],
            "keep_separate_pairs":resolution["keep_separate_pairs"],"unresolved_merge_candidates":len(resolution["unresolved_merge_candidates"]),
            "nodes_before_resolution":len(local_entities),"nodes":len(entities),"edges":len(edges),"connected_components":component_count(entities,edges),
            "isolated_nodes":isolated_node_ids(entities,edges),"neo4j":neo4j,"validation":validation,"embedding_calls":totals["embedding_calls"],"remote_model_calls":totals["primary_http_attempts"],
            "candidate_threshold":resolution["candidate_threshold"],"merge_threshold":resolution["merge_threshold"],"vector_dimension":resolution["vector_dimension"],"unique_pair_count":resolution["unique_pair_count"]}
