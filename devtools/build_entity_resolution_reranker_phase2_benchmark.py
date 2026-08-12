"""Expand curated Phase 2 definitions into a versioned benchmark and fixed split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from devtools.entity_resolution_reranker_phase2 import Phase2Benchmark, Phase2Case, assert_no_cluster_leakage, fixed_cluster_split

ROOT = Path(__file__).resolve().parents[1]


def build(definitions: dict) -> Phase2Benchmark:
    cases = []
    for cluster, canonical, alternate, second_alternate, entity_type, category, reason in definitions["positive_clusters"]:
        for index, right in enumerate((alternate, second_alternate), 1):
            cases.append(Phase2Case(
                case_id=f"merge_{cluster}_{index}", identity_cluster=cluster,
                source_artifact="reviewed project terminology and accepted fixtures",
                document_ids=[], graph_group=f"group_{cluster}",
                entity_a={"canonical_name": canonical, "entity_type": entity_type, "aliases": [alternate]},
                entity_b={"canonical_name": right, "entity_type": entity_type, "aliases": [canonical]},
                expected="MERGE", category=category, review_reason=reason,
            ))
    for case_id, left, right, entity_type, category in definitions["negative_pairs"]:
        cases.append(Phase2Case(
            case_id=f"split_{case_id}", identity_cluster=f"negative_{case_id}",
            source_artifact="reviewed formal-document hard-negative inventory",
            document_ids=[], graph_group=f"group_{case_id}",
            entity_a={"canonical_name": left, "entity_type": entity_type, "aliases": []},
            entity_b={"canonical_name": right, "entity_type": entity_type, "aliases": []},
            expected="SPLIT", category=category,
            review_reason=f"Reviewed distinct identities in category {category}.",
            deterministic_block=category in {"version_conflict", "opposition", "state_conflict", "role_conflict"},
        ))
    for case_id, left, right, entity_type, left_group, right_group in definitions["isolation_pairs"]:
        cases.append(Phase2Case(
            case_id=case_id, identity_cluster=case_id, source_artifact="graph-group isolation sanity fixture",
            document_ids=[], graph_group=f"{left_group}|{right_group}",
            entity_a={"canonical_name": left, "entity_type": entity_type, "aliases": [], "graph_group": left_group},
            entity_b={"canonical_name": right, "entity_type": entity_type, "aliases": [], "graph_group": right_group},
            expected="SPLIT", category="cross_graph_isolation", review_reason="Graph-group ownership forbids comparison.",
            isolation_only=True, deterministic_block=True,
        ))
    return Phase2Benchmark(benchmark_version="2", cases=cases)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "release_audit/reranker_phase2")
    args = parser.parse_args()
    definitions = json.loads((ROOT / "devtools/fixtures/entity_resolution_reranker_phase2_definitions.json").read_text(encoding="utf-8"))
    benchmark = build(definitions)
    split = fixed_cluster_split(benchmark.cases)
    assert_no_cluster_leakage(benchmark.cases, split)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "benchmark_v2.json").write_text(json.dumps(benchmark.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    (args.output / "split_manifest.json").write_text(json.dumps({"seed": 20260812, "partitions": split}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(benchmark.cases), "eligible": sum(not c.isolation_only for c in benchmark.cases),
                      "merge": sum(c.expected == "MERGE" and not c.isolation_only for c in benchmark.cases),
                      "split": sum(c.expected == "SPLIT" and not c.isolation_only for c in benchmark.cases)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
