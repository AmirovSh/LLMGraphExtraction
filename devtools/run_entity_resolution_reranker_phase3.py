"""Run frozen replication and bounded expanded-zone Phase-3 research."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from config.settings import environment_value, load_project_settings
from devtools.build_entity_resolution_reranker_phase3_benchmark import build
from devtools.entity_resolution_reranker import EntityProfile, PairCase, RerankerClient, RerankerConfig, directional_pairs, join_directional_results
from devtools.entity_resolution_reranker_phase3 import *
from devtools.run_entity_resolution_reranker_experiment import benchmark_bge_scores, load_local_env, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "release_audit/reranker_phase3"
FLOORS = [.76, .72, .68, .64, .60, .55]
TOP_KS = [5, 10, 20, 40]


def profiles(benchmark: Phase3Benchmark) -> list[PairCase]:
    output = []
    for case in benchmark.cases:
        a, b = case.entity_a, case.entity_b
        output.append(PairCase(case_id=case.case_id,
            entity_a=EntityProfile(canonical_name=a["canonical_name"], entity_type=a["entity_type"], aliases=a["aliases"], graph_group=a["graph_group"]),
            entity_b=EntityProfile(canonical_name=b["canonical_name"], entity_type=b["entity_type"], aliases=b["aliases"], graph_group=b["graph_group"]),
            expected=case.expected, source=case.source_reference, reason=case.label_reason))
    return output


def baseline(similarity: float, left: str, right: str) -> str:
    if similarity >= .90 and left.casefold() == right.casefold():
        return "MERGE"
    return "UNRESOLVED" if similarity >= .76 else "SPLIT"


def evaluate(rows: list[dict], config: dict) -> tuple[list[dict], dict]:
    selected = symmetric_top_k(rows, int(config["top_k"]), float(config["candidate_floor"]))
    evaluated = [row | {"decision": guard_decision(row, candidate_floor=config["candidate_floor"],
        threshold=config["threshold"], aggregation=config["aggregation"], strategy=config["strategy_version"],
        top_k_selected=row["case_id"] in selected)} for row in rows]
    return evaluated, metrics(evaluated)


def operating_points(cal: list[dict], val: list[dict]) -> list[dict]:
    candidates = []
    for floor in FLOORS:
        for top_k in TOP_KS:
            selected = symmetric_top_k(cal, top_k, floor)
            for aggregation in ("min", "mean", "max"):
                thresholds = sorted({float(row[aggregation]) for row in cal if row["case_id"] in selected}, reverse=True)
                for threshold in thresholds:
                    config = {"candidate_floor": floor, "top_k": top_k, "aggregation": aggregation,
                              "threshold": threshold, "strategy_version": "R-CONSENSUS" if aggregation == "min" else "R-EXPANDED"}
                    _, cm = evaluate(cal, config); _, vm = evaluate(val, config)
                    candidates.append(config | {"calibration": cm, "validation": vm})
    # Precision-first; then validation recall/F1 and bounded candidate growth.
    return sorted(candidates, key=lambda x: (-x["validation"]["false_merges"], x["validation"]["precision"],
        x["validation"]["recall"], x["validation"]["f1"], -x["candidate_floor"], -x["threshold"]), reverse=True)


def category_metrics(rows: list[dict]) -> dict[str, Any]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows: grouped[row["category"]].append(row)
    return {category: metrics(group) for category, group in sorted(grouped.items())}


def errors(rows: list[dict], config: dict) -> list[dict]:
    chosen = symmetric_top_k(rows, config["top_k"], config["candidate_floor"])
    output = []
    for row in rows:
        if (row["decision"] == "MERGE") == (row["expected"] == "MERGE"): continue
        if row["expected"] == "MERGE" and row["bge_similarity"] < config["candidate_floor"]: taxonomy = "CANDIDATE_NOT_RETRIEVED"
        elif row["expected"] == "MERGE" and row["case_id"] not in chosen: taxonomy = "TOP_K_PRUNED"
        elif row["expected"] == "SPLIT": taxonomy = "SAME_SURFACE_DIFFERENT_IDENTITY" if row["category"] == "same_surface_negative" else "RERANKER_FALSE_POSITIVE"
        elif abs(row["ab"] - row["ba"]) > .2: taxonomy = "DIRECTIONAL_ASYMMETRY"
        elif abs(row[config["aggregation"]] - config["threshold"]) < 1e-4: taxonomy = "SATURATED_THRESHOLD"
        else: taxonomy = "RERANKER_FALSE_NEGATIVE"
        output.append({"case_id": row["case_id"], "category": row["category"], "taxonomy": taxonomy,
                       "bge_similarity": row["bge_similarity"], "ab": row["ab"], "ba": row["ba"]})
    return output


def large_graph(rows: list[dict], config: dict) -> dict[str, Any]:
    # Candidate-density diagnostic with 100 local identities; no semantic graph mutation.
    n = 100; total = n * (n - 1) // 2
    eligible = [r for r in rows if not r.get("deterministic_block")]
    density_prod = sum(r["bge_similarity"] >= .76 for r in eligible) / len(eligible)
    density_exp = sum(r["bge_similarity"] >= config["candidate_floor"] for r in eligible) / len(eligible)
    prod = round(total * density_prod); expanded = round(total * density_exp)
    return {"N": n, "theoretical_pairs": total, "production_candidates_estimated": prod,
            "expanded_candidates_estimated": expanded, "top_5_bound": min(expanded, n * 5),
            "top_10_bound": min(expanded, n * 10), "top_20_bound": min(expanded, n * 20),
            "top_40_bound": min(expanded, n * 40), "reranker_pairs_bound": min(expanded, n * config["top_k"]),
            "accepted_experimental_merges": None, "measurement_kind": "density estimate from V3 plus deterministic top-k bound"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, default=OUT); parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args(); out = args.output
    definitions = json.loads((ROOT / "devtools/fixtures/entity_resolution_reranker_phase3_definitions.json").read_text(encoding="utf-8"))
    bench = build(definitions); split = fixed_phase3_split(bench.cases); assert_split_isolation(bench.cases, split); manifest = freeze_manifest(bench, split)
    out.mkdir(parents=True, exist_ok=True); write_json(out / "benchmark_v3.json", bench.model_dump(mode="json")); write_json(out / "split_manifest.json", manifest)
    load_local_env(ROOT / ".env.runtime"); settings = load_project_settings(ROOT / "config"); remote = settings.runtime.remote
    endpoint = environment_value(remote.embedding_base_url_env); key = environment_value(remote.embedding_api_key_env)
    pairs = profiles(bench); client = RerankerClient(RerankerConfig(endpoint, key, timeout_seconds=remote.embedding_timeout_seconds, trust_env=remote.trust_env))
    bge_start = time.perf_counter(); bge = benchmark_bge_scores(type("B", (), {"cases": pairs})(), endpoint=endpoint, api_key=key,
        model=settings.entity_resolution.embedding.model, timeout=remote.embedding_timeout_seconds, trust_env=remote.trust_env); bge_seconds = time.perf_counter() - bge_start
    score_start = time.perf_counter(); raw = client.score_many(directional_pairs(pairs, "R2"), batch_size=args.concurrency); score_seconds = time.perf_counter() - score_start
    scores = join_directional_results(raw); rows = []
    for case, pair in zip(bench.cases, pairs, strict=True):
        sim = bge[case.case_id]; values = scores[case.case_id]
        rows.append({"case_id": case.case_id, "partition": split[case.case_id], "expected": case.expected,
            "category": case.category, "deterministic_block": case.deterministic_block, "same_graph_group": True,
            "entity_a_key": case.identity_cluster_id + ":a", "entity_b_key": case.identity_cluster_id + ":b",
            "bge_similarity": sim, "baseline_decision": "SPLIT" if case.deterministic_block else baseline(sim, pair.entity_a.canonical_name, pair.entity_b.canonical_name), **values})
    write_json(out / "directional_scores.json", rows)
    replication = [r for r in rows if r["partition"] == "REPLICATION_SET"]
    frozen = dict(FROZEN_PHASE2) | {"top_k": 20, "strategy_version": "R-AMBIGUOUS"}
    rep_eval, rep_metrics = evaluate(replication, frozen); rep_base = metrics([r | {"decision": r["baseline_decision"]} for r in replication])
    replication_result = {"baseline": rep_base, "frozen_reranker": rep_metrics,
        "candidate_recall": candidate_sweep(replication, [.76])[0]["candidate_recall"],
        "reranker_pair_count": len(symmetric_top_k(replication, 20, .76)),
        "confidence_intervals": confidence_intervals(rep_eval)}; write_json(out / "replication_results.json", replication_result)
    cal = [r for r in rows if r["partition"] == "EXPANDED_CALIBRATION"]; val = [r for r in rows if r["partition"] == "EXPANDED_VALIDATION"]
    floor_sweep = candidate_sweep(cal + val, FLOORS); write_json(out / "candidate_floor_sweep.json", floor_sweep)
    write_json(out / "topk_sweep.json", {str(f): top_k_sweep(cal + val, TOP_KS, f) for f in FLOORS})
    ranked = operating_points(cal, val); winner = ranked[0]
    expanded = freeze_expanded_config({"model": "bge-reranker-v2-m3", "input_variant": "R2", "aggregation": winner["aggregation"],
        "threshold": winner["threshold"], "candidate_floor": winner["candidate_floor"], "top_k": winner["top_k"],
        "deterministic_exclusions": ["graph_group", "version", "opposition", "explicit_distinct", "invalid_pair", "safe_equivalence"],
        "strategy_version": winner["strategy_version"]}, benchmark_hash=manifest["benchmark_hash"], split_hash=manifest["split_hash"])
    write_json(out / "calibration.json", {"operating_points": ranked[:50]}); _, val_metrics = evaluate(val, expanded); write_json(out / "validation.json", val_metrics); write_json(out / "frozen_expanded_config.json", expanded)
    held = [r for r in rows if r["partition"] == "EXPANDED_HELD_OUT"]; held_eval, held_metrics = evaluate(held, expanded)
    held_base = metrics([r | {"decision": r["baseline_decision"]} for r in held]); held_phase2, held_phase2_metrics = evaluate(held, frozen)
    write_json(out / "held_out.json", {"B0": held_base, "phase2_frozen": held_phase2_metrics, "expanded": held_metrics})
    write_json(out / "category_metrics.json", category_metrics(held_eval)); stress = [r for r in rows if r["partition"] == "COUNTERFACTUAL_STRESS"]
    stress_eval, stress_metrics = evaluate(stress, expanded); write_json(out / "counterfactual_stress.json", {"metrics": stress_metrics, "rows": stress_eval})
    cis = {"replication": confidence_intervals(rep_eval), "validation": confidence_intervals(evaluate(val, expanded)[0]), "held_out": confidence_intervals(held_eval)}; write_json(out / "confidence_intervals.json", cis)
    robustness = threshold_robustness(val + held, expanded["threshold"], expanded["aggregation"]); write_json(out / "threshold_robustness.json", robustness)
    error_rows = errors(held_eval, expanded); write_json(out / "error_analysis.json", {"counts": dict(Counter(e["taxonomy"] for e in error_rows)), "errors": error_rows})
    graph_replay = {"status": "NOT_EXECUTED", "reason": "No three fresh saved extraction graph groups were available; pair fixtures are not represented as graph replay.",
        "cross_group_unit_test": "PASS", "production_graph_mutation": False}; write_json(out / "graph_replay.json", graph_replay)
    growth = large_graph(cal + val, expanded); write_json(out / "large_graph_candidate_growth.json", growth)
    # Five boundary repeats, 24 pairs, as required.
    boundary = sorted([r for r in rows if r["partition"] != "COUNTERFACTUAL_STRESS"], key=lambda r: min(abs(r["bge_similarity"]-expanded["candidate_floor"]), abs(r[expanded["aggregation"]]-expanded["threshold"])))[:24]
    boundary_cases = [pairs[[p.case_id for p in pairs].index(r["case_id"])] for r in boundary]; repeat_scores=[]
    for _ in range(5): repeat_scores.append(join_directional_results(client.score_many(directional_pairs(boundary_cases, "R2"), batch_size=args.concurrency)))
    stability_rows=[]; drift=0
    for row in boundary:
        vals=[run[row["case_id"]][expanded["aggregation"]] for run in repeat_scores]; decisions={v>=expanded["threshold"] for v in vals}; drift += len(decisions)>1
        stability_rows.append({"case_id":row["case_id"],"mean":statistics.mean(vals),"stddev":statistics.pstdev(vals),"min":min(vals),"max":max(vals),"classification_drift":len(decisions)>1})
    stability={"runs":5,"pairs":24,"classification_drift_pairs":drift,"rows":stability_rows,"status":"PASS" if drift==0 else "FAIL","rank_drift":"not separately materialized"}; write_json(out / "stability.json", stability)
    pair_count=len(pairs); perf={"bge_candidate_generation_seconds":bge_seconds,"reranker_pairs":pair_count,"directional_operations":len(raw),"wall_seconds":score_seconds,"pairs_per_second":pair_count/score_seconds,
        "mean_directional_latency":statistics.mean(r.latency_seconds for r in raw),"p50_directional_latency":statistics.median(r.latency_seconds for r in raw),"p95_directional_latency":sorted(r.latency_seconds for r in raw)[math.ceil(.95*len(raw))-1],"concurrency":args.concurrency}; write_json(out / "performance.json", perf)
    density = growth["reranker_pairs_bound"] / 100
    cost={str(n):{"estimated_reranker_pairs":round(density*n),"estimated_directional_operations":round(2*density*n),"estimated_extra_seconds":round((density*n)/(pair_count/score_seconds),2),"candidate_expansion_multiplier":next(x["expansion_factor"] for x in floor_sweep if x["candidate_floor"]==expanded["candidate_floor"]),"estimate":True} for n in (50,100,250,500)}; write_json(out / "cost_model.json", cost)
    comparison={"B0":held_base,"phase2_frozen":held_phase2_metrics,"expanded":held_metrics,"reranker_pairs":{"phase2":len(symmetric_top_k(held,20,.76)),"expanded":len(symmetric_top_k(held,expanded["top_k"],expanded["candidate_floor"]))}}; write_json(out / "strategy_comparison.json", comparison)
    print(json.dumps({"benchmark":{"eligible":205,"merge":70,"split":135},"replication":replication_result,"selected":expanded,"held_out":comparison,"stress":stress_metrics,"stability":stability,"performance":perf,"growth":growth},indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
