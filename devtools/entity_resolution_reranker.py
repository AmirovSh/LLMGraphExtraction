"""Isolated pairwise reranker experiment; never imported by production code."""
from __future__ import annotations

import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator


Aggregation = Literal["mean", "min", "max"]
InputVariant = Literal["R1", "R2"]


class EntityProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    canonical_name: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    graph_group: str = Field(min_length=1)


class PairCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1)
    entity_a: EntityProfile
    entity_b: EntityProfile
    expected: Literal["MERGE", "SPLIT"]
    source: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    explicit_equivalence: bool = False


class Benchmark(BaseModel):
    model_config = ConfigDict(extra="forbid")
    benchmark_version: str = Field(min_length=1)
    cases: list[PairCase] = Field(min_length=1)

    @field_validator("cases")
    @classmethod
    def unique_case_ids(cls, value: list[PairCase]) -> list[PairCase]:
        ids = [item.case_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("case_id values must be unique")
        return value


@dataclass(frozen=True)
class RerankerConfig:
    endpoint: str
    api_key: str
    model: str = "bge-reranker-v2-m3"
    timeout_seconds: float = 30.0
    trust_env: bool = False


@dataclass(frozen=True)
class DirectionalPair:
    pair_id: str
    query: str
    document: str
    direction: Literal["ab", "ba"]


@dataclass(frozen=True)
class RerankerResult:
    pair_id: str
    direction: Literal["ab", "ba"]
    score: float
    latency_seconds: float


def render_profile(entity: EntityProfile, variant: InputVariant) -> str:
    if variant == "R1":
        return entity.canonical_name
    aliases = "\n".join(f"- {alias}" for alias in entity.aliases) or "- none"
    return (
        f"canonical_name: {entity.canonical_name}\n"
        f"type: {entity.entity_type}\n"
        f"aliases:\n{aliases}"
    )


def directional_pairs(cases: Sequence[PairCase], variant: InputVariant) -> list[DirectionalPair]:
    output: list[DirectionalPair] = []
    for case in cases:
        left = render_profile(case.entity_a, variant)
        right = render_profile(case.entity_b, variant)
        output.extend((
            DirectionalPair(case.case_id, left, right, "ab"),
            DirectionalPair(case.case_id, right, left, "ba"),
        ))
    return output


def validate_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("reranker score must be numeric")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError("reranker score must be finite")
    return score


class RerankerClient:
    def __init__(self, config: RerankerConfig, *, post: Callable[..., Any] = httpx.post) -> None:
        self.config = config
        self.post = post

    def _score(self, pair: DirectionalPair) -> RerankerResult:
        started = time.perf_counter()
        response = self.post(
            f"{self.config.endpoint.rstrip('/')}/rerank",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            json={"model": self.config.model, "query": pair.query, "documents": [pair.document]},
            timeout=self.config.timeout_seconds,
            trust_env=self.config.trust_env,
        )
        response.raise_for_status()
        body = response.json()
        results = body.get("results")
        if not isinstance(results, list) or len(results) != 1 or int(results[0].get("index", -1)) != 0:
            raise ValueError("reranker response must contain exactly index 0")
        return RerankerResult(
            pair.pair_id, pair.direction, validate_score(results[0].get("relevance_score")),
            time.perf_counter() - started,
        )

    def score_many(self, pairs: Sequence[DirectionalPair], *, batch_size: int = 1) -> list[RerankerResult]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        indexed = list(enumerate(pairs))
        output: list[tuple[int, RerankerResult]] = []
        for offset in range(0, len(indexed), batch_size):
            chunk = indexed[offset:offset + batch_size]
            if len(chunk) == 1:
                output.append((chunk[0][0], self._score(chunk[0][1])))
            else:
                with ThreadPoolExecutor(max_workers=len(chunk)) as executor:
                    results = list(executor.map(lambda row: self._score(row[1]), chunk))
                output.extend((row[0], result) for row, result in zip(chunk, results, strict=True))
        return [item for _, item in sorted(output)]


def aggregate_scores(score_ab: float, score_ba: float) -> dict[str, float]:
    left, right = validate_score(score_ab), validate_score(score_ba)
    return {"mean": (left + right) / 2, "min": min(left, right), "max": max(left, right)}


def join_directional_results(results: Iterable[RerankerResult]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    for item in results:
        if item.direction in grouped.setdefault(item.pair_id, {}):
            raise ValueError("duplicate directional reranker result")
        grouped[item.pair_id][item.direction] = item.score
    for pair_id, directions in grouped.items():
        if set(directions) != {"ab", "ba"}:
            raise ValueError(f"missing direction for {pair_id}")
        directions.update(aggregate_scores(directions["ab"], directions["ba"]))
    return grouped


def production_decision(
    case: PairCase, bge_similarity: float, *, candidate_threshold: float, merge_threshold: float,
) -> Literal["MERGE", "SPLIT", "UNRESOLVED", "INELIGIBLE"]:
    if case.entity_a.graph_group != case.entity_b.graph_group:
        return "INELIGIBLE"
    if case.explicit_equivalence:
        return "MERGE"
    left, right = case.entity_a.canonical_name.strip(), case.entity_b.canonical_name.strip()
    if bge_similarity < candidate_threshold:
        return "SPLIT"
    if bge_similarity < merge_threshold:
        return "UNRESOLVED"
    if left.casefold() == right.casefold():
        return "MERGE"
    return "UNRESOLVED"


def strategy_decision(
    baseline: str, reranker_score: float, *, strategy: Literal["B0", "R-GUARD", "R-AMBIGUOUS"],
    threshold: float,
) -> str:
    score = validate_score(reranker_score)
    if baseline == "INELIGIBLE" or strategy == "B0":
        return baseline
    if strategy == "R-GUARD":
        return "SPLIT" if baseline == "MERGE" and score < threshold else baseline
    if baseline == "UNRESOLVED":
        return "MERGE" if score >= threshold else "SPLIT"
    return baseline


def classification_metrics(rows: Sequence[Mapping[str, Any]], decision_key: str) -> dict[str, Any]:
    eligible = [row for row in rows if row[decision_key] != "INELIGIBLE"]
    tp = sum(row[decision_key] == "MERGE" and row["expected"] == "MERGE" for row in eligible)
    fp = sum(row[decision_key] == "MERGE" and row["expected"] == "SPLIT" for row in eligible)
    fn = sum(row[decision_key] != "MERGE" and row["expected"] == "MERGE" for row in eligible)
    tn = sum(row[decision_key] != "MERGE" and row["expected"] == "SPLIT" for row in eligible)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"true_merges": tp, "true_splits": tn, "false_merges": fp, "missed_merges": fn,
            "precision": precision, "recall": recall, "f1": f1, "eligible_pairs": len(eligible)}


def threshold_sweep(
    rows: Sequence[Mapping[str, Any]], *, aggregation: Aggregation, strategy: Literal["R-GUARD", "R-AMBIGUOUS"],
) -> list[dict[str, Any]]:
    scores = sorted({float(row[aggregation]) for row in rows if row["baseline_decision"] != "INELIGIBLE"})
    thresholds = sorted({scores[0] - 1e-9, *scores, scores[-1] + 1e-9}) if scores else []
    output = []
    for threshold in thresholds:
        evaluated = [dict(row) | {"decision": strategy_decision(
            str(row["baseline_decision"]), float(row[aggregation]), strategy=strategy, threshold=threshold,
        )} for row in rows]
        output.append({"strategy": strategy, "aggregation": aggregation, "threshold": threshold,
                       **classification_metrics(evaluated, "decision")})
    return output


def select_precision_first(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("threshold sweep is empty")
    return max(rows, key=lambda row: (
        -int(row["false_merges"]), float(row["precision"]), float(row["recall"]),
        float(row["f1"]), -int(row["missed_merges"]), float(row["threshold"]),
    ))


def performance_summary(results: Sequence[RerankerResult], wall_seconds: float, entity_pairs: int) -> dict[str, Any]:
    latencies = sorted(item.latency_seconds for item in results)
    percentile = lambda p: latencies[min(len(latencies) - 1, math.ceil(p * len(latencies)) - 1)]
    return {
        "entity_pairs": entity_pairs, "directional_scoring_operations": len(results),
        "wall_seconds": wall_seconds, "pairs_per_second": entity_pairs / wall_seconds if wall_seconds else None,
        "latency_mean_seconds": statistics.mean(latencies) if latencies else None,
        "latency_p50_seconds": percentile(0.50) if latencies else None,
        "latency_p95_seconds": percentile(0.95) if latencies else None,
    }


def assert_group_isolation(cases: Sequence[PairCase], candidate_ids: Iterable[str]) -> None:
    by_id = {case.case_id: case for case in cases}
    cross_group = [case_id for case_id in candidate_ids
                   if by_id[case_id].entity_a.graph_group != by_id[case_id].entity_b.graph_group]
    if cross_group:
        raise ValueError(f"cross-graph-group candidates detected: {cross_group}")


def apply_guard(pair_decisions: Sequence[Mapping[str, Any]], allowed_merge_pairs: set[tuple[str, str]]) -> list[dict[str, Any]]:
    """Demote merges only; never create a candidate or alter non-merge decisions."""
    output = []
    for item in pair_decisions:
        row = dict(item)
        key = tuple(sorted((str(row["left_local_id"]), str(row["right_local_id"]))))
        if row["decision"] == "MERGE" and key not in allowed_merge_pairs:
            row["decision"] = "KEEP_SEPARATE"
            row["reason"] = "experimental reranker precision guard"
        output.append(row)
    return output
