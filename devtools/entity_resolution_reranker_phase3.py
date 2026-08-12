"""Offline contracts and analysis for the isolated Phase-3 reranker study.

This module is intentionally unreachable from ``runtime``.  It models research
data and decisions only; production entity resolution remains authoritative.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from devtools.entity_resolution_reranker import validate_score


FROZEN_PHASE2 = {
    "model": "bge-reranker-v2-m3",
    "input_variant": "R2",
    "aggregation": "max",
    "production_floor": 0.76,
    "candidate_floor": 0.76,
    "threshold": 0.9999665021896362,
    "strategy": "R-AMBIGUOUS",
}
PARTITIONS = ("REPLICATION_SET", "EXPANDED_CALIBRATION", "EXPANDED_VALIDATION", "EXPANDED_HELD_OUT")


class Phase3Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1)
    identity_cluster_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    entity_a: dict[str, Any]
    entity_b: dict[str, Any]
    expected: Literal["MERGE", "SPLIT"]
    label_reason: str = Field(min_length=1)
    source_kind: str = Field(min_length=1)
    source_reference: str = Field(min_length=1)
    deterministic_block: str | None = None
    diagnostic_only: bool = False


class Phase3Benchmark(BaseModel):
    model_config = ConfigDict(extra="forbid")
    benchmark_version: Literal["3.1"]
    cases: list[Phase3Case] = Field(min_length=180)

    @field_validator("cases")
    @classmethod
    def validate_cases(cls, cases: list[Phase3Case]) -> list[Phase3Case]:
        ids = [case.case_id for case in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate Phase-3 case_id")
        return cases


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def fixed_phase3_split(cases: Sequence[Phase3Case], *, seed: int = 20260813) -> dict[str, str]:
    """Freeze cluster-isolated tracks without consulting any model score."""
    primary = [case for case in cases if not case.diagnostic_only]
    clusters = sorted({case.identity_cluster_id for case in primary})
    rng = random.Random(seed)
    rng.shuffle(clusters)
    replication_n = max(1, round(len(clusters) * 0.34))
    remaining = len(clusters) - replication_n
    calibration_n = round(remaining * 0.60)
    validation_n = round(remaining * 0.20)
    mapping: dict[str, str] = {}
    for index, cluster in enumerate(clusters):
        if index < replication_n:
            mapping[cluster] = "REPLICATION_SET"
        elif index < replication_n + calibration_n:
            mapping[cluster] = "EXPANDED_CALIBRATION"
        elif index < replication_n + calibration_n + validation_n:
            mapping[cluster] = "EXPANDED_VALIDATION"
        else:
            mapping[cluster] = "EXPANDED_HELD_OUT"
    return {case.case_id: ("COUNTERFACTUAL_STRESS" if case.diagnostic_only else mapping[case.identity_cluster_id])
            for case in cases}


def assert_split_isolation(cases: Sequence[Phase3Case], split: Mapping[str, str]) -> None:
    seen: dict[str, str] = {}
    for case in cases:
        partition = split[case.case_id]
        previous = seen.setdefault(case.identity_cluster_id, partition)
        if previous != partition:
            raise ValueError(f"identity cluster leaked: {case.identity_cluster_id}")


def freeze_manifest(benchmark: Phase3Benchmark, split: Mapping[str, str], *, seed: int = 20260813) -> dict[str, Any]:
    data = benchmark.model_dump(mode="json")
    counts = {name: sum(part == name for part in split.values()) for name in (*PARTITIONS, "COUNTERFACTUAL_STRESS")}
    categories: dict[str, int] = defaultdict(int)
    classes: dict[str, int] = defaultdict(int)
    for case in benchmark.cases:
        categories[case.category] += 1
        classes[case.expected] += 1
    manifest = {"seed": seed, "benchmark_hash": canonical_json_hash(data), "partitions": dict(split),
                "identity_clusters": sorted({c.identity_cluster_id for c in benchmark.cases}),
                "partition_counts": counts, "class_counts": dict(classes), "category_counts": dict(categories)}
    return manifest | {"split_hash": canonical_json_hash(manifest)}


def guard_decision(row: Mapping[str, Any], *, candidate_floor: float, threshold: float,
                   aggregation: Literal["min", "mean", "max"], strategy: str,
                   top_k_selected: bool = True) -> str:
    """Experimental decision: ownership/conflict gates always outrank score."""
    if not row.get("same_graph_group", True):
        return "INELIGIBLE"
    if row.get("deterministic_block"):
        return "SPLIT"
    baseline = str(row.get("baseline_decision", "SPLIT"))
    if baseline == "MERGE":
        return "MERGE"
    similarity = float(row["bge_similarity"])
    if similarity < candidate_floor or not top_k_selected:
        return "SPLIT"
    if strategy == "R-AMBIGUOUS" and similarity < FROZEN_PHASE2["production_floor"]:
        return "SPLIT"
    ab, ba = validate_score(row["ab"]), validate_score(row["ba"])
    score = {"min": min(ab, ba), "mean": (ab + ba) / 2, "max": max(ab, ba)}[aggregation]
    return "MERGE" if score >= threshold else "SPLIT"


def metrics(rows: Sequence[Mapping[str, Any]], decision_key: str = "decision") -> dict[str, Any]:
    eligible = [r for r in rows if r[decision_key] != "INELIGIBLE"]
    tp = sum(r[decision_key] == "MERGE" and r["expected"] == "MERGE" for r in eligible)
    fp = sum(r[decision_key] == "MERGE" and r["expected"] == "SPLIT" for r in eligible)
    fn = sum(r[decision_key] != "MERGE" and r["expected"] == "MERGE" for r in eligible)
    tn = sum(r[decision_key] != "MERGE" and r["expected"] == "SPLIT" for r in eligible)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "false_merges": fp, "missed_merges": fn}


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> dict[str, float]:
    if trials == 0:
        return {"lower": 0.0, "upper": 1.0}
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return {"lower": max(0.0, centre - margin), "upper": min(1.0, centre + margin)}


def confidence_intervals(rows: Sequence[Mapping[str, Any]], *, decision_key: str = "decision",
                         resamples: int = 2000, seed: int = 20260813) -> dict[str, Any]:
    result = metrics(rows, decision_key)
    predicted = result["tp"] + result["fp"]
    positives = result["tp"] + result["fn"]
    negatives = result["fp"] + result["tn"]
    rng = random.Random(seed)
    f1s = []
    for _ in range(resamples):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        f1s.append(metrics(sample, decision_key)["f1"])
    f1s.sort()
    return {"seed": seed, "resamples": resamples,
            "precision": wilson_interval(result["tp"], predicted),
            "recall": wilson_interval(result["tp"], positives),
            "f1_bootstrap": {"lower": f1s[math.floor(.025 * (resamples - 1))],
                             "upper": f1s[math.ceil(.975 * (resamples - 1))]},
            "false_merge_rate": wilson_interval(result["fp"], negatives)}


def candidate_sweep(rows: Sequence[Mapping[str, Any]], floors: Sequence[float], *, production_floor: float = .76) -> list[dict[str, Any]]:
    positives = sum(r["expected"] == "MERGE" for r in rows)
    baseline = sum(float(r["bge_similarity"]) >= production_floor and not r.get("deterministic_block") for r in rows)
    output = []
    for floor in floors:
        selected = [r for r in rows if float(r["bge_similarity"]) >= floor and not r.get("deterministic_block")]
        pos = sum(r["expected"] == "MERGE" for r in selected)
        entities = {str(r["entity_a_key"]) for r in selected} | {str(r["entity_b_key"]) for r in selected}
        output.append({"candidate_floor": floor, "candidate_pairs": len(selected), "retrieved_merge_pairs": pos,
                       "retrieved_split_pairs": len(selected) - pos, "candidate_recall": pos / positives if positives else 0.0,
                       "candidate_precision": pos / len(selected) if selected else 0.0,
                       "expansion_factor": len(selected) / baseline if baseline else None,
                       "pairs_per_entity": len(selected) / len(entities) if entities else 0.0})
    return output


def symmetric_top_k(rows: Sequence[Mapping[str, Any]], top_k: int, floor: float) -> set[str]:
    incident: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if float(row["bge_similarity"]) >= floor and not row.get("deterministic_block"):
            incident[str(row["entity_a_key"])].append(row)
            incident[str(row["entity_b_key"])].append(row)
    chosen: set[str] = set()
    for candidates in incident.values():
        ranked = sorted(candidates, key=lambda r: (-float(r["bge_similarity"]), str(r["case_id"])))
        chosen.update(str(r["case_id"]) for r in ranked[:top_k])
    return chosen


def top_k_sweep(rows: Sequence[Mapping[str, Any]], top_ks: Sequence[int], floor: float) -> list[dict[str, Any]]:
    positives = {str(r["case_id"]) for r in rows if r["expected"] == "MERGE" and float(r["bge_similarity"]) >= floor}
    output = []
    for top_k in top_ks:
        chosen = symmetric_top_k(rows, top_k, floor)
        output.append({"top_k": top_k, "candidate_volume": len(chosen), "reranker_pairs": len(chosen),
                       "candidate_recall": len(chosen & positives) / len(positives) if positives else 0.0,
                       "positive_candidates_lost": len(positives - chosen)})
    return output


def threshold_robustness(rows: Sequence[Mapping[str, Any]], threshold: float, aggregation: str,
                         epsilons: Sequence[float] = (1e-3, 1e-4, 1e-5)) -> dict[str, Any]:
    base = [validate_score(r[aggregation]) >= threshold for r in rows]
    probes = []
    for epsilon in epsilons:
        for sign in (-1, 1):
            probe = threshold + sign * epsilon
            decisions = [validate_score(r[aggregation]) >= probe for r in rows]
            probes.append({"threshold": probe, "epsilon": sign * epsilon,
                           "classification_changes": sum(a != b for a, b in zip(base, decisions, strict=True))})
    near = {str(e): sum(abs(validate_score(r[aggregation]) - threshold) <= e for r in rows) for e in epsilons}
    return {"threshold": threshold, "near_threshold": near, "probes": probes}


def assert_frozen_phase2(config: Mapping[str, Any]) -> None:
    for key, expected in FROZEN_PHASE2.items():
        if config.get(key) != expected:
            raise ValueError(f"Phase-2 frozen field changed: {key}")


def freeze_expanded_config(config: Mapping[str, Any], *, benchmark_hash: str, split_hash: str) -> dict[str, Any]:
    required = {"model", "input_variant", "aggregation", "threshold", "candidate_floor", "top_k",
                "deterministic_exclusions", "strategy_version"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"missing frozen config fields: {sorted(missing)}")
    frozen = dict(config) | {"benchmark_hash": benchmark_hash, "split_hash": split_hash}
    return frozen | {"config_hash": canonical_json_hash(frozen)}
