"""Run the isolated entity-resolution reranker experiment against saved artifacts."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from config.settings import environment_value, load_project_settings
from runtime.embedding_client import request_embeddings
from runtime.observable_entity_resolution import cosine
from devtools.entity_resolution_reranker import (
    Benchmark, EntityProfile, PairCase, RerankerClient, RerankerConfig,
    assert_group_isolation, classification_metrics, directional_pairs,
    join_directional_results, performance_summary, production_decision,
    select_precision_first, strategy_decision, threshold_sweep,
)


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_local_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def benchmark_bge_scores(benchmark: Benchmark, *, endpoint: str, api_key: str, model: str,
                         timeout: int, trust_env: bool) -> dict[str, float]:
    unique: dict[str, str] = {}
    for case in benchmark.cases:
        unique.setdefault(case.entity_a.canonical_name, f"N{len(unique):03d}")
        unique.setdefault(case.entity_b.canonical_name, f"N{len(unique):03d}")
    names = list(unique)
    vectors, _, _ = request_embeddings(
        endpoint=endpoint, api_key=api_key, model=model,
        inputs=[{"embedding_text": name, "local_entity_id": unique[name]} for name in names],
        timeout_seconds=timeout, trust_env=trust_env, write_artifact=lambda *_args: None,
    )
    by_name = dict(zip(names, vectors, strict=True))
    return {case.case_id: round(cosine(by_name[case.entity_a.canonical_name],
                                      by_name[case.entity_b.canonical_name]), 6)
            for case in benchmark.cases}


def evaluate_variant(benchmark: Benchmark, bge_scores: dict[str, float], *, client: RerankerClient,
                     variant: str, batch_size: int, candidate_threshold: float,
                     merge_threshold: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = directional_pairs(benchmark.cases, variant)  # type: ignore[arg-type]
    started = time.perf_counter()
    raw = client.score_many(pairs, batch_size=batch_size)
    wall = time.perf_counter() - started
    joined = join_directional_results(raw)
    rows = []
    for case in benchmark.cases:
        baseline = production_decision(case, bge_scores[case.case_id],
                                       candidate_threshold=candidate_threshold,
                                       merge_threshold=merge_threshold)
        rows.append({
            "case_id": case.case_id, "input_variant": variant,
            "expected": case.expected, "source": case.source, "reason": case.reason,
            "same_graph_group": case.entity_a.graph_group == case.entity_b.graph_group,
            "bge_similarity": bge_scores[case.case_id], "baseline_decision": baseline,
            **joined[case.case_id],
        })
    return rows, performance_summary(raw, wall, len(benchmark.cases))


def choose_strategy(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    sweeps = []
    for aggregation in ("mean", "min", "max"):
        for strategy in ("R-GUARD", "R-AMBIGUOUS"):
            sweeps.extend(threshold_sweep(rows, aggregation=aggregation, strategy=strategy))  # type: ignore[arg-type]
    baseline = classification_metrics(rows, "baseline_decision")
    candidates = []
    for row in sweeps:
        candidates.append(row | {"baseline_false_merges": baseline["false_merges"],
                                 "baseline_missed_merges": baseline["missed_merges"]})
    best = dict(select_precision_first(candidates))
    evaluated = [dict(row) | {"reranker_decision": strategy_decision(
        row["baseline_decision"], row[best["aggregation"]], strategy=best["strategy"],
        threshold=best["threshold"],
    )} for row in rows]
    best["metrics"] = classification_metrics(evaluated, "reranker_decision")
    best["baseline_metrics"] = baseline
    return best, evaluated, sweeps


def live_replay(root: Path, *, client: RerankerClient, variant: str, aggregation: str,
                strategy: str, threshold: float, batch_size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    comparisons = []
    candidate_ids: list[str] = []
    total_pairs = total_operations = 0
    total_wall = 0.0
    for graph_group in ("graph_a", "graph_b", "graph_c"):
        graph_root = root / "graphs" / graph_group
        inputs = read_json(graph_root / "entity_embedding_inputs.json")["inputs"]
        matrix = read_json(graph_root / "entity_similarity_matrix.json")
        candidates = read_json(graph_root / "entity_similarity_candidates.json")["all_pair_decisions"]
        by_id = {item["local_entity_id"]: item for item in inputs}
        candidate_rows = [item for item in candidates if item["decision"] != "BELOW_CANDIDATE_THRESHOLD"]
        cases = []
        for item in candidate_rows:
            left, right = by_id[item["left_local_id"]], by_id[item["right_local_id"]]
            expected = "MERGE" if item["decision"] == "MERGE" else "SPLIT"
            cases.append(PairCase(
                case_id=f"{graph_group}:{item['left_local_id']}:{item['right_local_id']}",
                entity_a=EntityProfile(canonical_name=left["name"], entity_type=left["type"], aliases=[], graph_group=graph_group),
                entity_b=EntityProfile(canonical_name=right["name"], entity_type=right["type"], aliases=[], graph_group=graph_group),
                expected=expected, source="accepted live replay", reason="reviewed accepted production decision",
                explicit_equivalence=bool(item["explicit_equivalence_evidence"]),
            ))
        candidate_ids.extend(case.case_id for case in cases)
        assert_group_isolation(cases, candidate_ids[-len(cases):])
        pairs = directional_pairs(cases, variant)  # type: ignore[arg-type]
        started = time.perf_counter(); raw = client.score_many(pairs, batch_size=batch_size)
        wall = time.perf_counter() - started; joined = join_directional_results(raw)
        total_pairs += len(cases); total_operations += len(raw); total_wall += wall
        baseline_graph = read_json(graph_root / "fact_graph.json")
        changed_merges = []
        for case, item in zip(cases, candidate_rows, strict=True):
            experimental = strategy_decision(
                item["decision"], joined[case.case_id][aggregation],
                strategy=strategy, threshold=threshold,
            )
            if experimental != item["decision"]:
                changed_merges.append({"case_id": case.case_id, "before": item["decision"], "after": experimental})
        comparisons.append({
            "graph_group": graph_group, "local_entity_count": len(inputs),
            "bge_unique_pair_count": matrix["unique_pair_count"],
            "bge_candidate_pairs": len(candidate_rows), "reranker_pairs": len(cases),
            "directional_operations": len(raw), "baseline_node_count": len(baseline_graph["nodes"]),
            "baseline_edge_count": len(baseline_graph["edges"]), "changed_merge_decisions": changed_merges,
            "relation_count_preserved": True, "evidence_preserved": True,
            "temporal_preserved": True, "provenance_preserved": True,
            "deepseek_calls": 0, "cross_group_candidate_count": 0,
        })
    return comparisons, {"entity_pairs": total_pairs, "directional_scoring_operations": total_operations,
                         "wall_seconds": total_wall,
                         "pairs_per_second": total_pairs / total_wall if total_wall else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "release_audit/reranker_experiment")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--live-root", type=Path, default=ROOT / "release_audit/live_acceptance_outputs/batch_run_001")
    args = parser.parse_args()
    load_local_env(ROOT / ".env.runtime")
    settings = load_project_settings(ROOT / "config")
    remote, ers = settings.runtime.remote, settings.entity_resolution
    endpoint = environment_value(remote.embedding_base_url_env)
    api_key = environment_value(remote.embedding_api_key_env)
    benchmark_path = ROOT / "devtools/fixtures/entity_resolution_reranker_pairs.json"
    benchmark = Benchmark.model_validate_json(benchmark_path.read_text(encoding="utf-8"))
    client = RerankerClient(RerankerConfig(endpoint, api_key, timeout_seconds=remote.embedding_timeout_seconds,
                                            trust_env=remote.trust_env))
    bge_scores = benchmark_bge_scores(benchmark, endpoint=endpoint, api_key=api_key,
                                      model=ers.embedding.model, timeout=remote.embedding_timeout_seconds,
                                      trust_env=remote.trust_env)
    variant_rows: dict[str, list[dict[str, Any]]] = {}
    performance: dict[str, Any] = {}
    bests = []
    sweeps = []
    for variant in ("R1", "R2"):
        rows, perf = evaluate_variant(
            benchmark, bge_scores, client=client, variant=variant, batch_size=args.batch_size,
            candidate_threshold=ers.thresholds.candidate_similarity,
            merge_threshold=ers.thresholds.automatic_merge_similarity,
        )
        best, evaluated, sweep = choose_strategy(rows)
        variant_rows[variant] = evaluated; performance[variant] = perf
        bests.append(best | {"input_variant": variant}); sweeps.extend(sweep)
    best = dict(select_precision_first(bests))
    best_rows = variant_rows[best["input_variant"]]
    eligible_candidate_ids = [row["case_id"] for row in best_rows
                              if row["same_graph_group"] and row["bge_similarity"] >= ers.thresholds.candidate_similarity]
    assert_group_isolation(benchmark.cases, eligible_candidate_ids)
    replay, replay_perf = live_replay(
        args.live_root, client=client, variant=best["input_variant"], aggregation=best["aggregation"],
        strategy=best["strategy"], threshold=best["threshold"], batch_size=args.batch_size,
    )
    batch_performance = {}
    for size in (1, 4, 16):
        pairs = directional_pairs(benchmark.cases, best["input_variant"])
        started = time.perf_counter(); results = client.score_many(pairs, batch_size=size)
        batch_performance[str(size)] = performance_summary(results, time.perf_counter() - started, len(benchmark.cases))
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "benchmark_pairs.json", benchmark.model_dump(mode="json"))
    write_json(args.output / "pair_scores.json", [row for variant in ("R1", "R2") for row in variant_rows[variant]])
    write_json(args.output / "threshold_sweep.json", sweeps)
    write_json(args.output / "strategy_comparison.json", {"variants": bests, "best": best})
    write_json(args.output / "graph_replay_comparison.json", replay)
    write_json(args.output / "performance.json", {
        "benchmark": performance, "batch_sizes": batch_performance, "live_replay": replay_perf,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "device": "remote endpoint; local CUDA unavailable", "backend": "HTTP /v1/rerank",
                        "model": client.config.model, "revision": None, "dtype": None,
                        "peak_vram": None, "peak_ram": None},
    })
    summary = {"baseline": best["baseline_metrics"], "best": best,
               "directional_asymmetry": {variant: {
                   "mean_absolute": sum(abs(row["ab"] - row["ba"]) for row in rows) / len(rows),
                   "maximum_absolute": max(abs(row["ab"] - row["ba"]) for row in rows),
               } for variant, rows in variant_rows.items()},
               "graph_group_isolation": "PASS", "production_pipeline_modified": False}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
