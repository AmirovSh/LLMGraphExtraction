"""Finalize the real-artifact reranker study when its dataset gate fails."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ARTIFACTS = (
    "benchmark_queries.json", "candidate_generation.json", "reranker_ranking.json",
    "ranking_metrics.json", "category_metrics.json", "abbreviation_analysis.json",
    "cross_document_analysis.json", "multiple_distractor_analysis.json",
    "catastrophic_demotions.json", "stability.json", "graph_replay.json",
    "large_graph_replay.json", "performance.json", "verifier_budget.json",
)


def finalize(inventory_path: Path, report_path: Path) -> None:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory["dataset_gate"] != "INSUFFICIENT_DATA":
        raise ValueError("finalizer is only valid for a failed dataset gate")
    output = inventory_path.parent
    reason = (
        f"Only {inventory['maximum_bidirectional_identity_queries']} source-grounded real queries "
        f"are available; minimum is {inventory['minimum_identity_queries']}."
    )
    for name in ARTIFACTS:
        payload = {
            "status": "NOT_RUN",
            "dataset_gate": "INSUFFICIENT_DATA",
            "reason": reason,
            "production_modified": False,
            "reranker_calls": 0,
        }
        (output / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    runs = inventory["runs"]
    observed = sum(run["explicit_equivalence_observations"] for run in runs)
    multi = sum(run["document_count"] > 1 for run in runs)
    report_path.write_text(f"""# Reranker Real-Artifact Shortlisting Report

## Verdict

**RERANKER REAL-ARTIFACT SHORTLISTING PARTIAL — INSUFFICIENT_DATA**

The mandatory dataset gate failed. Saved real artifacts provide only
{inventory['unique_source_grounded_identity_pairs']} unique identity pairs, or at most
{inventory['maximum_bidirectional_identity_queries']} bidirectional queries, below the required minimum of
{inventory['minimum_identity_queries']}. The {observed} observations include repeated executions of the same
source evidence and therefore cannot be counted independently.

No synthetic examples were added. Automatic MERGE decisions, BGE scores, reranker scores,
and current graph outcomes were not used as ground truth. No endpoint was called after the
gate failed.

## Inventory

- saved reports inspected: {len(runs)}
- multi-document reports with document metadata: {multi}
- unique source-grounded identity pairs: {inventory['unique_source_grounded_identity_pairs']}
- maximum identity-match queries: {inventory['maximum_bidirectional_identity_queries']}
- required identity-match queries: {inventory['minimum_identity_queries']}
- dataset gate: `INSUFFICIENT_DATA`

The inventory includes per-run graph group, document count, local entity count, candidate
pair count, explicit-equivalence count, and cross-document indication. A large saved graph
exists, but graph size cannot repair the absence of independent identity labels.

## Ranking and replay

All ranking, stability, distractor, abbreviation, cross-document, verifier-budget,
performance, and graph-replay measurements are `NOT_RUN`. Reporting Recall@K from fewer
than 50 real queries would violate the task gate. Accordingly there are no top-3/top-5
retention claims, no catastrophic-demotion claims, and no empirical reranker cost claim.

Production behavior was not modified. The reranker caused zero merges because it was not
invoked. Existing graphs, relations, evidence, and provenance were not rewritten.

## Recommendation

**Recommended top-K: NONE.** First create a reviewed, source-grounded real benchmark with
at least 50 identity-match queries, including multi-document aliases and multiple related
distractors. Then rerun the unchanged ranking-only protocol. The later architectural path
may still be `candidate generation -> reranker top-K -> identity verifier`, but this dataset
cannot justify choosing top-3 or top-5.
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.inventory, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
