"""Isolated, pre-resolution identity verifier experiment primitives."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["SAME_ENTITY", "DIFFERENT_ENTITY", "UNCERTAIN"]
Representation = Literal["V1", "V2", "V3", "V4", "V4_LITE"]
FORBIDDEN = frozenset({"identity_cluster_id", "label", "expected", "candidate_score",
                       "resolution_decision", "merge_result", "final_graph", "neo4j"})


class VerifierDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Decision
    reason_codes: list[str] = Field(max_length=6)
    evidence_refs: list[str] = Field(max_length=8)


@dataclass(frozen=True)
class EntityProfile:
    local_id: str
    canonical_name: str
    entity_type: str
    aliases: tuple[str, ...] = ()
    evidence: tuple[tuple[str, str], ...] = ()
    relations: tuple[str, ...] = ()


def _clean(value: str) -> str:
    return " ".join(value.split())


def render_profile(profile: EntityProfile, representation: Representation, *, relation_limit: int | None = None) -> dict[str, Any]:
    lines = [_clean(profile.canonical_name)]
    if representation != "V1":
        lines = [f"canonical_name: {_clean(profile.canonical_name)}",
                 f"type: {_clean(profile.entity_type)}", "aliases:"]
        lines.extend(f"- {_clean(x)}" for x in profile.aliases[:8] if _clean(x))
        if lines[-1] == "aliases:": lines.append("- none")
    if representation in {"V3", "V4", "V4_LITE"}:
        lines.append("evidence:")
        for ref, text in profile.evidence[:5]: lines.append(f"- [{ref}] {_clean(text)[:160]}")
        if lines[-1] == "evidence:": lines.append("- none")
    if representation in {"V4", "V4_LITE"}:
        lines.append("observed_local_relations:")
        limit = relation_limit if relation_limit is not None else (8 if representation == "V4" else 3)
        lines.extend(f"- {_clean(x)[:160]}" for x in profile.relations[:limit])
        if lines[-1] == "observed_local_relations:": lines.append("- none")
    text = "\n".join(lines)[:800]
    assert_pre_resolution({"local_id": profile.local_id, "representation": representation, "text": text})
    return {"local_id": profile.local_id, "representation": representation, "text": text,
            "sha256": hashlib.sha256(text.encode()).hexdigest(), "characters": len(text),
            "raw_relation_count": len(profile.relations), "retained_relation_count": min(len(profile.relations), limit) if representation in {"V4", "V4_LITE"} else 0,
            "raw_characters": len("\n".join(lines)), "truncated": len("\n".join(lines)) > 800,
            "approx_tokens": max(1, (len(text)+3)//4)}
def assert_pre_resolution(payload: Any) -> None:
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).casefold() in FORBIDDEN: raise ValueError(f"forbidden field: {key}")
                walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    walk(payload)


def verifier_tool() -> dict[str, Any]:
    return {"type":"function","function":{"name":"verify_entity_identity",
        "description":"Return one conservative identity decision.","parameters":{"type":"object",
        "properties":{"decision":{"type":"string","enum":["SAME_ENTITY","DIFFERENT_ENTITY","UNCERTAIN"]},
        "reason_codes":{"type":"array","items":{"type":"string"},"maxItems":6},
        "evidence_refs":{"type":"array","items":{"type":"string"},"maxItems":8}},
        "required":["decision","reason_codes","evidence_refs"],"additionalProperties":False}}}


SYSTEM_PROMPT = """You are a conservative entity identity verifier. Decide whether Profile A and Profile B denote the same real named entity. SAME_ENTITY requires positive identity evidence. DIFFERENT_ENTITY requires positive incompatibility or clear related-but-distinct evidence. Otherwise return UNCERTAIN. Type equality, lexical similarity, co-occurrence, and relatedness are not sufficient. Use only supplied profiles. Call verify_entity_identity exactly once and return no prose."""


def deterministic_baseline(left: EntityProfile, right: EntityProfile) -> Decision:
    norm = lambda s: re.sub(r"[^a-z0-9]+", "", s.casefold())
    if norm(left.canonical_name) == norm(right.canonical_name): return "SAME_ENTITY"
    return "UNCERTAIN"


def metrics(rows: list[dict[str, Any]], prediction_key: str = "decision") -> dict[str, Any]:
    labels = ("SAME_ENTITY", "DIFFERENT_ENTITY", "UNCERTAIN")
    matrix = {a:{b:0 for b in labels} for a in labels}
    for row in rows: matrix[row["label"]][row[prediction_key]] += 1
    same_tp=matrix["SAME_ENTITY"]["SAME_ENTITY"]
    same_fp=matrix["DIFFERENT_ENTITY"]["SAME_ENTITY"]
    same_fn=sum(matrix["SAME_ENTITY"][x] for x in labels if x!="SAME_ENTITY")
    diff_tp=matrix["DIFFERENT_ENTITY"]["DIFFERENT_ENTITY"]
    diff_fp=matrix["SAME_ENTITY"]["DIFFERENT_ENTITY"]
    diff_fn=sum(matrix["DIFFERENT_ENTITY"][x] for x in labels if x!="DIFFERENT_ENTITY")
    div=lambda a,b: a/b if b else 0.0
    decided=sum(row[prediction_key]!="UNCERTAIN" for row in rows)
    correct=sum(row[prediction_key]==row["label"] for row in rows)
    return {"n":len(rows),"confusion_matrix":matrix,"accuracy":div(correct,len(rows)),
        "coverage":div(decided,len(rows)),"selective_accuracy":div(correct,decided),
        "same_precision":div(same_tp,same_tp+same_fp),"same_recall":div(same_tp,same_tp+same_fn),
        "different_precision":div(diff_tp,diff_tp+diff_fp),"different_recall":div(diff_tp,diff_tp+diff_fn),
        "false_merges":same_fp,"false_splits":diff_fp,"uncertain":len(rows)-decided}


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
