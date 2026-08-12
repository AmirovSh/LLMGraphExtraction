"""Held-out calibration utilities for the isolated reranker experiment."""
from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from typing import Any, Iterable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from devtools.entity_resolution_reranker import validate_score


Partition = Literal["CALIBRATION", "VALIDATION", "HELD_OUT_TEST", "ISOLATION"]


class Phase2Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1)
    identity_cluster: str = Field(min_length=1)
    source_artifact: str = Field(min_length=1)
    document_ids: list[str]
    graph_group: str = Field(min_length=1)
    entity_a: dict[str, Any]
    entity_b: dict[str, Any]
    expected: Literal["MERGE", "SPLIT"]
    category: str = Field(min_length=1)
    review_reason: str = Field(min_length=1)
    isolation_only: bool = False
    deterministic_block: bool = False


class Phase2Benchmark(BaseModel):
    model_config = ConfigDict(extra="forbid")
    benchmark_version: Literal["2"]
    cases: list[Phase2Case] = Field(min_length=100)

    @field_validator("cases")
    @classmethod
    def unique_ids(cls, cases: list[Phase2Case]) -> list[Phase2Case]:
        ids = [case.case_id for case in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate Phase 2 case_id")
        return cases


def fixed_cluster_split(cases: Sequence[Phase2Case], *, seed: int = 20260812) -> dict[str, str]:
    """Assign whole semantic clusters once; isolation cases never enter metrics."""
    clusters = sorted({case.identity_cluster for case in cases if not case.isolation_only})
    rng = random.Random(seed)
    rng.shuffle(clusters)
    calibration_end = round(len(clusters) * 0.60)
    validation_end = calibration_end + round(len(clusters) * 0.20)
    cluster_partition = {
        cluster: ("CALIBRATION" if index < calibration_end else
                  "VALIDATION" if index < validation_end else "HELD_OUT_TEST")
        for index, cluster in enumerate(clusters)
    }
    return {case.case_id: ("ISOLATION" if case.isolation_only else cluster_partition[case.identity_cluster])
            for case in cases}


def assert_no_cluster_leakage(cases: Sequence[Phase2Case], split: Mapping[str, str]) -> None:
    seen: dict[str, str] = {}
    for case in cases:
        partition = split[case.case_id]
        if partition == "ISOLATION":
            continue
        previous = seen.setdefault(case.identity_cluster, partition)
        if previous != partition:
            raise ValueError(f"identity cluster leaked across partitions: {case.identity_cluster}")


def candidate_sweep(rows: Sequence[Mapping[str, Any]], floors: Sequence[float], *, production_floor: float) -> list[dict[str, Any]]:
    eligible = [row for row in rows if not row.get("isolation_only") and not row.get("deterministic_block")]
    positives = sum(row["expected"] == "MERGE" for row in eligible)
    baseline_count = sum(float(row["bge_similarity"]) >= production_floor for row in eligible)
    output = []
    for floor in floors:
        selected = [row for row in eligible if float(row["bge_similarity"]) >= floor]
        positive = sum(row["expected"] == "MERGE" for row in selected)
        negative = len(selected) - positive
        output.append({
            "candidate_floor": floor, "candidate_pairs": len(selected),
            "positive_pairs_retrieved": positive, "negative_pairs_retrieved": negative,
            "candidate_recall": positive / positives if positives else 0.0,
            "candidate_precision": positive / len(selected) if selected else 0.0,
            "candidate_expansion_factor": len(selected) / baseline_count if baseline_count else None,
        })
    return output


def top_k_sweep(rows: Sequence[Mapping[str, Any]], top_ks: Sequence[int], *, floor: float) -> list[dict[str, Any]]:
    """Apply symmetric per-entity top-k over the labelled candidate graph."""
    eligible = [row for row in rows if not row.get("isolation_only") and not row.get("deterministic_block")
                and float(row["bge_similarity"]) >= floor]
    positives_total = sum(row["expected"] == "MERGE" for row in rows
                          if not row.get("isolation_only") and not row.get("deterministic_block"))
    output = []
    for top_k in top_ks:
        selected_ids: set[str] = set()
        incident: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in eligible:
            incident[str(row["entity_a_key"])].append(row)
            incident[str(row["entity_b_key"])].append(row)
        for candidates in incident.values():
            ranked = sorted(candidates, key=lambda item: (float(item["bge_similarity"]), str(item["case_id"])), reverse=True)
            selected_ids.update(str(item["case_id"]) for item in ranked[:top_k])
        selected = [row for row in eligible if str(row["case_id"]) in selected_ids]
        positives = sum(row["expected"] == "MERGE" for row in selected)
        output.append({"top_k": top_k, "candidate_pairs": len(selected), "reranker_pairs": len(selected),
                       "candidate_recall": positives / positives_total if positives_total else 0.0})
    return output


def expanded_decision(row: Mapping[str, Any], *, production_floor: float, candidate_floor: float,
                      threshold: float, aggregation: str) -> str:
    """Experimental-only decision subordinate to ownership and deterministic blocks."""
    if row.get("isolation_only") or not row.get("same_graph_group", True):
        return "INELIGIBLE"
    if row.get("deterministic_block"):
        return "SPLIT"
    baseline = str(row["baseline_decision"])
    if baseline == "MERGE":
        return "MERGE"
    similarity = float(row["bge_similarity"])
    if similarity < candidate_floor:
        return "SPLIT"
    score = validate_score(row[aggregation])
    if similarity >= production_floor and baseline != "UNRESOLVED":
        return baseline
    return "MERGE" if score >= threshold else "SPLIT"


def metrics(rows: Sequence[Mapping[str, Any]], decision_key: str) -> dict[str, Any]:
    eligible = [row for row in rows if row[decision_key] != "INELIGIBLE"]
    tp = sum(row[decision_key] == "MERGE" and row["expected"] == "MERGE" for row in eligible)
    fp = sum(row[decision_key] == "MERGE" and row["expected"] == "SPLIT" for row in eligible)
    fn = sum(row[decision_key] != "MERGE" and row["expected"] == "MERGE" for row in eligible)
    unresolved = sum(row[decision_key] == "UNRESOLVED" for row in eligible)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "false_merges": fp,
            "missed_merges": fn, "true_merges": tp, "unresolved_pairs": unresolved,
            "eligible_pairs": len(eligible)}


def calibrate_without_held_out(rows: Sequence[Mapping[str, Any]], *, production_floor: float,
                               floors: Sequence[float]) -> dict[str, Any]:
    if any(row["partition"] == "HELD_OUT_TEST" for row in rows):
        raise ValueError("held-out labels must not enter calibration")
    calibration = [row for row in rows if row["partition"] == "CALIBRATION"]
    validation = [row for row in rows if row["partition"] == "VALIDATION"]
    candidates = []
    for variant in ("R1", "R2"):
        variant_cal = [row for row in calibration if row["input_variant"] == variant]
        variant_val = [row for row in validation if row["input_variant"] == variant]
        for aggregation in ("mean", "min", "max"):
            observed = sorted({float(row[aggregation]) for row in variant_cal})
            for floor in floors:
                for threshold in observed:
                    cal_eval = [dict(row) | {"decision": expanded_decision(
                        row, production_floor=production_floor, candidate_floor=floor,
                        threshold=threshold, aggregation=aggregation)} for row in variant_cal]
                    val_eval = [dict(row) | {"decision": expanded_decision(
                        row, production_floor=production_floor, candidate_floor=floor,
                        threshold=threshold, aggregation=aggregation)} for row in variant_val]
                    cal_metrics, val_metrics = metrics(cal_eval, "decision"), metrics(val_eval, "decision")
                    candidates.append({"input_variant": variant, "aggregation": aggregation,
                                       "candidate_floor": floor, "threshold": threshold,
                                       "calibration": cal_metrics, "validation": val_metrics})
    return max(candidates, key=lambda row: (
        -row["validation"]["false_merges"], row["validation"]["precision"],
        row["validation"]["recall"], row["validation"]["f1"],
        -row["calibration"]["false_merges"], row["candidate_floor"], row["threshold"],
    ))


def bootstrap_intervals(rows: Sequence[Mapping[str, Any]], decision_key: str, *,
                        resamples: int = 2000, seed: int = 20260812) -> dict[str, Any]:
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {"precision": [], "recall": [], "f1": []}
    for _ in range(resamples):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        result = metrics(sample, decision_key)
        for key in samples:
            samples[key].append(float(result[key]))
    intervals = {}
    for key, values in samples.items():
        values.sort()
        intervals[key] = {"lower": values[math.floor(0.025 * (resamples - 1))],
                          "upper": values[math.ceil(0.975 * (resamples - 1))]}
    return {"seed": seed, "resamples": resamples, "intervals": intervals}


def logit(score: float, epsilon: float = 1e-9) -> float:
    bounded = min(1 - epsilon, max(epsilon, validate_score(score)))
    return math.log(bounded / (1 - bounded))


def fingerprint_config(config: Mapping[str, Any]) -> str:
    encoded = repr(sorted(config.items())).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
