"""Inventory source-grounded identity evidence in saved real graph artifacts.

This is a dataset-gate utility, not an entity resolver.  Automatic resolution
decisions and similarity scores are deliberately excluded from ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

MINIMUM_IDENTITY_QUERIES = 50


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _local_entity_count(directory: Path) -> int:
    path = directory / "local_entities.json"
    if path.exists():
        value = _read_json(path)
        return len(value) if isinstance(value, list) else 0
    path = directory / "entity_embedding_inputs.json"
    if path.exists():
        value = _read_json(path)
        if isinstance(value, list):
            return len(value)
        if isinstance(value, dict):
            for key in ("entities", "inputs", "items"):
                if isinstance(value.get(key), list):
                    return len(value[key])
    return 0


def _manifest_documents(directory: Path, report: dict[str, Any]) -> list[str]:
    manifest = directory / "graph_manifest.json"
    if manifest.exists():
        value = _read_json(manifest)
        if isinstance(value, dict):
            return [str(item) for item in _as_list(value.get("document_ids"))]
    return [str(item) for item in _as_list(report.get("source_document_ids"))]


def _evidence_key(pair: dict[str, Any]) -> str:
    relation_ids = sorted(str(item) for item in _as_list(pair.get("equivalence_relation_ids")))
    span_ids = sorted(str(item) for item in _as_list(pair.get("equivalence_evidence_span_ids")))
    payload = {"relation_ids": relation_ids, "span_ids": span_ids}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def inventory_reports(roots: Iterable[Path]) -> dict[str, Any]:
    report_paths: set[Path] = set()
    for root in roots:
        if root.exists():
            report_paths.update(root.rglob("entity_resolution.json"))
            report_paths.update(root.rglob("resolution_report.json"))

    runs: list[dict[str, Any]] = []
    unique_evidence: dict[str, dict[str, Any]] = {}
    for path in sorted(report_paths):
        value = _read_json(path)
        if not isinstance(value, dict):
            continue
        candidates = _as_list(value.get("candidate_pairs"))
        explicit = [
            pair for pair in candidates
            if isinstance(pair, dict) and pair.get("explicit_equivalence_evidence") is True
        ]
        documents = _manifest_documents(path.parent, value)
        for pair in explicit:
            key = _evidence_key(pair)
            # Evidence-less flags cannot establish independent source-grounded truth.
            if key == _evidence_key({}):
                continue
            unique_evidence.setdefault(key, {
                "left_local_id": pair.get("left_local_id"),
                "right_local_id": pair.get("right_local_id"),
                "equivalence_relation_ids": _as_list(pair.get("equivalence_relation_ids")),
                "equivalence_evidence_span_ids": _as_list(pair.get("equivalence_evidence_span_ids")),
                "observed_in": [],
            })["observed_in"].append(str(path))
        runs.append({
            "run_id": str(path.parent),
            "graph_group": path.parent.name,
            "documents": documents,
            "document_count": len(documents),
            "local_entity_count": _local_entity_count(path.parent),
            "potential_query_entities": len(candidates),
            "explicit_equivalence_observations": len(explicit),
            "manually_defensible_identity_matches": len({
                _evidence_key(pair) for pair in explicit if _evidence_key(pair) != _evidence_key({})
            }),
            "cross_document_aliases": len(documents) > 1 and bool(explicit),
        })

    query_count = len(unique_evidence) * 2
    return {
        "benchmark_kind": "REAL_ARTIFACT",
        "ground_truth_policy": "explicit source relation/evidence only; automatic merge and scores excluded",
        "minimum_identity_queries": MINIMUM_IDENTITY_QUERIES,
        "runs": runs,
        "unique_source_grounded_identity_pairs": len(unique_evidence),
        "maximum_bidirectional_identity_queries": query_count,
        "dataset_gate": "PASS" if query_count >= MINIMUM_IDENTITY_QUERIES else "INSUFFICIENT_DATA",
        "unique_evidence": list(unique_evidence.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = inventory_reports(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: inventory[key] for key in (
        "unique_source_grounded_identity_pairs", "maximum_bidirectional_identity_queries", "dataset_gate"
    )}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
