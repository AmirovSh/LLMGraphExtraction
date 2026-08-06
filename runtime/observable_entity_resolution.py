"""Observable embedding-only entity resolution for validated extraction runs."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from runtime.embedding_client import request_embeddings

_EXPLICIT_EQUIVALENCE_RELATIONS = frozenset({
    "alias",
    "aliases",
    "also_called",
    "also_known_as",
    "denotes_same_as",
    "equivalent_to",
    "refers_to",
    "same_as",
    "synonymous_with",
})


def cosine(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        raise ValueError("embedding vector has zero norm")
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _incident_context(entities: list[dict[str, Any]], relations: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {
        entity["local_id"]: {"relation_families": set(), "relation_descriptions": set(), "neighbour_types": set()}
        for entity in entities
    }
    by_id = {entity["local_id"]: entity for entity in entities}
    for relation in relations:
        left_id = relation["source_local_id"]
        right_id = relation["target_local_id"]
        if left_id not in contexts or right_id not in contexts:
            continue
        family = relation["relation_family"]
        contexts[left_id]["relation_families"].add(family)
        contexts[right_id]["relation_families"].add(family)
        contexts[left_id]["relation_descriptions"].add(relation["relation_description"])
        contexts[right_id]["relation_descriptions"].add(relation["relation_description"])
        contexts[left_id]["neighbour_types"].add(by_id[right_id]["type"])
        contexts[right_id]["neighbour_types"].add(by_id[left_id]["type"])
    return contexts


def _relation_key(value: object) -> str:
    normalized = []
    previous_separator = False
    for character in str(value).casefold().strip():
        if character.isalnum():
            normalized.append(character); previous_separator = False
        elif normalized and not previous_separator:
            normalized.append("_"); previous_separator = True
    return "".join(normalized).strip("_")


def _explicit_equivalence_evidence(
    entities: list[dict[str, Any]], relations: Iterable[dict[str, Any]],
) -> dict[frozenset[str], dict[str, list[str]]]:
    """Index only structurally extracted, evidence-backed equivalence relations."""
    entity_ids = {entity["local_id"] for entity in entities}
    evidence: dict[frozenset[str], dict[str, list[str]]] = {}
    for index, relation in enumerate(relations, start=1):
        left_id = relation.get("source_local_id")
        right_id = relation.get("target_local_id")
        spans = relation.get("evidence_span_ids") or []
        if (
            left_id not in entity_ids
            or right_id not in entity_ids
            or left_id == right_id
            or not spans
            or _relation_key(relation.get("raw_relation")) not in _EXPLICIT_EQUIVALENCE_RELATIONS
        ):
            continue
        pair = frozenset((left_id, right_id))
        item = evidence.setdefault(pair, {"relation_ids": [], "evidence_span_ids": []})
        relation_id = relation.get("relation_id") or relation.get("edge_id") or (
            f"{relation.get('unit_id', 'LOCAL')}:R{index:03d}"
        )
        item["relation_ids"].append(str(relation_id))
        item["evidence_span_ids"].extend(str(span_id) for span_id in spans)
    for item in evidence.values():
        item["relation_ids"] = sorted(set(item["relation_ids"]))
        item["evidence_span_ids"] = sorted(set(item["evidence_span_ids"]))
    return evidence


def _direct_distinct_relation_pairs(
    entities: list[dict[str, Any]], relations: Iterable[dict[str, Any]],
) -> set[frozenset[str]]:
    entity_ids = {entity["local_id"] for entity in entities}
    pairs: set[frozenset[str]] = set()
    for relation in relations:
        left_id = relation.get("source_local_id")
        right_id = relation.get("target_local_id")
        if (
            left_id in entity_ids
            and right_id in entity_ids
            and left_id != right_id
            and relation.get("evidence_span_ids")
            and _relation_key(relation.get("raw_relation")) not in _EXPLICIT_EQUIVALENCE_RELATIONS
        ):
            pairs.add(frozenset((left_id, right_id)))
    return pairs


def _confirmed_semantic_conflict(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Use only typed, upstream semantic attributes; never inspect surface wording."""
    conflicts: list[str] = []
    for field in ("stable_identifier", "semantic_state", "semantic_version"):
        left_value = left.get(field)
        right_value = right.get(field)
        if left_value is not None and right_value is not None and left_value != right_value:
            conflicts.append(field)
    left_attributes = left.get("semantic_attributes") or {}
    right_attributes = right.get("semantic_attributes") or {}
    if isinstance(left_attributes, Mapping) and isinstance(right_attributes, Mapping):
        for key in sorted(set(left_attributes) & set(right_attributes)):
            if left_attributes[key] != right_attributes[key]:
                conflicts.append(f"semantic_attributes.{key}")
    return bool(conflicts), conflicts


def _normalized_name_tokens(value: str) -> tuple[str, ...]:
    """Return language-neutral alphanumeric tokens for structural name comparison."""
    tokens: list[str] = []
    current: list[str] = []
    for character in value.casefold().strip():
        if character.isalnum():
            current.append(character)
        elif current:
            tokens.append("".join(current)); current = []
    if current: tokens.append("".join(current))
    return tuple(tokens)


def _contains_contiguous_tokens(container: tuple[str, ...], contained: tuple[str, ...]) -> bool:
    return bool(contained) and len(container) > len(contained) and any(
        container[index:index + len(contained)] == contained for index in range(len(container) - len(contained) + 1)
    )


def name_relation(left_name: str, right_name: str) -> str:
    if left_name.strip() == right_name.strip(): return "EXACT"
    if left_name.strip().casefold() == right_name.strip().casefold(): return "CASE_ONLY"
    left_tokens = _normalized_name_tokens(left_name); right_tokens = _normalized_name_tokens(right_name)
    if _contains_contiguous_tokens(left_tokens, right_tokens) or _contains_contiguous_tokens(right_tokens, left_tokens):
        return "TOKEN_CONTAINMENT_EXPANSION"
    return "DIFFERENT"


def _structural_decision(
    left: dict[str, Any], right: dict[str, Any], similarity: float, *, candidate_threshold: float,
    merge_threshold: float, contexts: dict[str, dict[str, Any]], decisions: Mapping[str, bool],
    equivalence_evidence: Mapping[frozenset[str], Mapping[str, list[str]]],
    direct_distinct_pairs: set[frozenset[str]],
) -> dict[str, Any]:
    left_context = contexts[left["local_id"]]
    right_context = contexts[right["local_id"]]
    type_match = left["type"] == right["type"]
    different_units = left["unit_id"] != right["unit_id"]
    left_name = left["name"].strip()
    right_name = right["name"].strip()
    exact_name_match = left_name == right_name
    case_only_name_match = not exact_name_match and left_name.casefold() == right_name.casefold()
    semantic_conflict, conflict_fields = _confirmed_semantic_conflict(left, right)
    relation = name_relation(left_name, right_name)
    explicit_evidence = equivalence_evidence.get(frozenset((left["local_id"], right["local_id"])))
    explicit_equivalence = explicit_evidence is not None
    direct_distinct_relation = frozenset((left["local_id"], right["local_id"])) in direct_distinct_pairs
    shared_relation_families = sorted(left_context["relation_families"] & right_context["relation_families"])
    shared_neighbour_types = sorted(left_context["neighbour_types"] & right_context["neighbour_types"])
    name_context = different_units and (
        (exact_name_match and decisions["exact_name_cross_unit_merge"])
        or (case_only_name_match and decisions["case_only_name_merge"])
    )
    structural_context = bool(shared_relation_families) or bool(shared_neighbour_types)
    context_compatible = name_context or structural_context
    base = {
        "left_local_id": left["local_id"],
        "right_local_id": right["local_id"],
        "similarity": round(similarity, 6),
        "left_type": left["type"],
        "right_type": right["type"],
        "type_match": type_match,
        "type_used_as_hard_filter": False,
        "name_relation": relation,
        "automatic_merge_blocked_by_name_expansion": False,
        "explicit_equivalence_evidence": explicit_equivalence,
        "equivalence_relation_ids": list((explicit_evidence or {}).get("relation_ids", [])),
        "equivalence_evidence_span_ids": list((explicit_evidence or {}).get("evidence_span_ids", [])),
        "merge_signal": "EXPLICIT_EQUIVALENCE" if explicit_equivalence else None,
        "direct_distinct_relation_evidence": direct_distinct_relation,
        "context_compatible": context_compatible,
        "different_units": different_units,
        "left_incident_relation_families": sorted(left_context["relation_families"]),
        "right_incident_relation_families": sorted(right_context["relation_families"]),
        "shared_incident_relation_families": shared_relation_families,
        "shared_neighbour_types": shared_neighbour_types,
        "semantic_conflict": semantic_conflict,
        "semantic_conflict_fields": conflict_fields,
    }
    if semantic_conflict and decisions["semantic_conflict_blocks_merge"]:
        return {**base, "decision": "KEEP_SEPARATE", "reason": "confirmed typed semantic conflict"}
    if direct_distinct_relation and not explicit_equivalence:
        return {**base, "decision": "KEEP_SEPARATE", "reason": "direct evidence-backed semantic relation distinguishes endpoints"}
    if explicit_equivalence:
        return {**base, "decision": "MERGE", "reason": "evidence-backed explicit equivalence relation"}
    if similarity < candidate_threshold:
        return {**base, "decision": "BELOW_CANDIDATE_THRESHOLD", "reason": "similarity is below candidate_threshold"}
    if similarity < merge_threshold:
        return {**base, "decision": "UNRESOLVED", "reason": "similarity is between candidate_threshold and merge_threshold"}
    if relation == "TOKEN_CONTAINMENT_EXPANSION" and not explicit_equivalence:
        return {**base, "automatic_merge_blocked_by_name_expansion": True, "decision": "UNRESOLVED", "reason": "automatic merge blocked by strict token-contained name expansion"}
    if decisions["require_context_for_semantic_alias_merge"] and not context_compatible:
        return {**base, "decision": "UNRESOLVED", "reason": "high similarity, but available context is insufficient for deterministic merge"}
    return {**base, "decision": "MERGE", "reason": "above merge_threshold with sufficient structural context"}


def evaluate_similarity(
    entities: list[dict[str, Any]], vectors: list[list[float]], relations: Iterable[dict[str, Any]], *,
    candidate_threshold: float, merge_threshold: float, decisions: Mapping[str, bool],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(entities) != len(vectors):
        raise ValueError("embedding response cardinality does not match entity inputs")
    if not 0 <= candidate_threshold <= merge_threshold <= 1:
        raise ValueError("thresholds must satisfy 0 <= candidate_threshold <= merge_threshold <= 1")
    relation_list = list(relations)
    contexts = _incident_context(entities, relation_list)
    equivalence_evidence = _explicit_equivalence_evidence(entities, relation_list)
    direct_distinct_pairs = _direct_distinct_relation_pairs(entities, relation_list)
    matrix: list[list[float]] = []
    pair_decisions: list[dict[str, Any]] = []
    for left_index, left_vector in enumerate(vectors):
        row: list[float] = []
        for right_index, right_vector in enumerate(vectors):
            value = round(cosine(left_vector, right_vector), 6)
            row.append(value)
            if right_index > left_index:
                pair_decisions.append(_structural_decision(
                    entities[left_index], entities[right_index], value,
                    candidate_threshold=candidate_threshold, merge_threshold=merge_threshold, contexts=contexts,
                    decisions=decisions, equivalence_evidence=equivalence_evidence,
                    direct_distinct_pairs=direct_distinct_pairs,
                ))
        matrix.append(row)
    return {
        "retrieval_method": "full_pairwise_cosine_matrix",
        "entity_count": len(entities),
        "unique_pair_count": len(pair_decisions),
        "entity_local_ids": [entity["local_id"] for entity in entities],
        "matrix": matrix,
        "candidate_threshold": candidate_threshold,
        "merge_threshold": merge_threshold,
    }, pair_decisions


def evaluate_saved_matrix(
    entities: list[dict[str, Any]], saved_matrix: Mapping[str, Any], relations: Iterable[dict[str, Any]], *,
    candidate_threshold: float, merge_threshold: float, decisions: Mapping[str, bool],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Re-evaluate a persisted small-run cosine matrix without embedding calls."""
    expected_ids = [entity["local_id"] for entity in entities]
    if saved_matrix.get("entity_local_ids") != expected_ids:
        raise ValueError("saved similarity matrix does not correspond to local entities")
    matrix = saved_matrix.get("matrix")
    if not isinstance(matrix, list) or len(matrix) != len(entities) or any(len(row) != len(entities) for row in matrix):
        raise ValueError("saved similarity matrix has invalid dimensions")
    relation_list = list(relations)
    contexts = _incident_context(entities, relation_list)
    equivalence_evidence = _explicit_equivalence_evidence(entities, relation_list)
    direct_distinct_pairs = _direct_distinct_relation_pairs(entities, relation_list)
    pair_decisions: list[dict[str, Any]] = []
    for left_index, left in enumerate(entities):
        for right_index in range(left_index + 1, len(entities)):
            pair_decisions.append(_structural_decision(
                left, entities[right_index], float(matrix[left_index][right_index]),
                candidate_threshold=candidate_threshold, merge_threshold=merge_threshold,
                contexts=contexts, decisions=decisions, equivalence_evidence=equivalence_evidence,
                direct_distinct_pairs=direct_distinct_pairs,
            ))
    return {
        "retrieval_method": saved_matrix.get("retrieval_method"),
        "entity_count": len(entities),
        "unique_pair_count": len(pair_decisions),
        "entity_local_ids": expected_ids,
        "matrix": matrix,
        "candidate_threshold": candidate_threshold,
        "merge_threshold": merge_threshold,
        "replayed_from_saved_matrix": True,
    }, pair_decisions


def resolve_pair_decisions(entities: list[dict[str, Any]], pair_decisions: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]]]:
    """Apply only already-recorded pair decisions; this function never calls a model."""
    parent = {entity["local_id"]: entity["local_id"] for entity in entities}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for item in pair_decisions:
        if item["decision"] == "MERGE":
            parent[find(item["right_local_id"])] = find(item["left_local_id"])
    global_ids: dict[str, str] = {}
    ids: dict[str, str] = {}
    for entity in entities:
        root = find(entity["local_id"])
        global_ids.setdefault(root, f"E{len(global_ids) + 1:03d}")
        ids[entity["local_id"]] = global_ids[root]
    candidates = [item for item in pair_decisions if item["decision"] != "BELOW_CANDIDATE_THRESHOLD"]
    return ids, {
        "candidate_pairs": candidates,
        "merged_pairs": [item for item in candidates if item["decision"] == "MERGE"],
        "keep_separate_pairs": [item for item in candidates if item["decision"] == "KEEP_SEPARATE"],
        "unresolved_merge_candidates": [item for item in candidates if item["decision"] == "UNRESOLVED"],
    }


def resolve_entities(
    entities: list[dict[str, Any]], relations: list[dict[str, Any]], *, endpoint: str, api_key: str, model: str,
    candidate_threshold: float, merge_threshold: float, decision_settings: Mapping[str, bool], pairwise_limit: int,
    top_k: int, timeout_seconds: int,
    trust_env: bool, write_artifact: Callable[[str, Any], None],
) -> tuple[dict[str, str], dict[str, Any]]:
    inputs = [
        {
            "local_entity_id": entity["local_id"], "name": entity["name"], "type": entity["type"],
            "evidence_span_ids": entity["evidence_span_ids"], "unit_id": entity["unit_id"],
            "embedding_text": entity["name"],
        }
        for entity in entities
    ]
    vectors, body, dimension = request_embeddings(
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        inputs=inputs,
        timeout_seconds=timeout_seconds,
        trust_env=trust_env,
        write_artifact=write_artifact,
    )
    matrix, pair_decisions = evaluate_similarity(
        entities, vectors, relations, candidate_threshold=candidate_threshold, merge_threshold=merge_threshold, decisions=decision_settings,
    )
    if len(entities) > pairwise_limit:
        # Preserve the evidence matrix for audit, while only exposing the configured nearest candidates.
        selected_pairs: set[tuple[str, str]] = set()
        ids = matrix["entity_local_ids"]
        for left_index, left_id in enumerate(ids):
            ranked = sorted(
                ((matrix["matrix"][left_index][right_index], ids[right_index]) for right_index in range(len(ids)) if right_index != left_index),
                reverse=True,
            )[:top_k]
            selected_pairs.update(tuple(sorted((left_id, right_id))) for _, right_id in ranked)
        pair_decisions = [
            item for item in pair_decisions
            if tuple(sorted((item["left_local_id"], item["right_local_id"]))) in selected_pairs
            or item["explicit_equivalence_evidence"]
        ]
        matrix["retrieval_method"] = "top_k_from_full_matrix"
        matrix["top_k"] = top_k
    write_artifact("entity_similarity_matrix.json", matrix)
    candidate_pairs = [item for item in pair_decisions if item["decision"] != "BELOW_CANDIDATE_THRESHOLD"]
    write_artifact("entity_similarity_candidates.json", {
        "retrieval_method": matrix["retrieval_method"], "top_k": matrix.get("top_k"),
        "candidate_threshold": candidate_threshold, "merge_threshold": merge_threshold,
        "all_pair_decisions": pair_decisions, "candidate_pairs": candidate_pairs,
    })
    ids, grouped = resolve_pair_decisions(entities, pair_decisions)
    counts = defaultdict(int)
    for item in candidate_pairs:
        counts[item["decision"]] += 1
    report = {
        "embedder_calls": 1, "usage": body.get("usage") or {},
        "retrieval_method": matrix["retrieval_method"], "top_k": matrix.get("top_k"),
        "candidate_threshold": candidate_threshold, "merge_threshold": merge_threshold,
        **grouped,
        "all_pair_decision_counts": dict(sorted(counts.items())),
        "vector_dimension": dimension,
        "embedding_input_count": len(inputs),
        "unique_pair_count": matrix["unique_pair_count"],
    }
    return ids, report
