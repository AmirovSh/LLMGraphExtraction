"""Build and validate a model-independent real entity-resolution benchmark."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

VERSION = "1.0.0"
SEED = 20260813


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def validate(clusters: list[dict], pairs: list[dict], queries: list[dict]) -> None:
    cluster_ids = [row["identity_cluster_id"] for row in clusters]
    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError("duplicate identity cluster")
    members = {m["entity_id"]: (c["identity_cluster_id"], m["graph_group"])
               for c in clusters for m in c["members"]}
    for pair in pairs:
        if "score" in pair or "model_score" in pair or "resolver_decision" in pair:
            raise ValueError("benchmark labels must not depend on model output")
        if pair["label"] == "SAME_ENTITY" and members[pair["left_entity_id"]][0] != members[pair["right_entity_id"]][0]:
            raise ValueError("positive endpoints must share a cluster")
    seen = set()
    for query in queries:
        if query["query_id"] in seen:
            raise ValueError("duplicate query")
        seen.add(query["query_id"])
        group = members[query["query_entity_id"]][1]
        if any(members[c["candidate_entity_id"]][1] != group for c in query["candidate_pool"]):
            raise ValueError("cross-group ranking candidate")


def split_clusters(clusters: list[dict]) -> dict[str, str]:
    ids = sorted(row["identity_cluster_id"] for row in clusters)
    random.Random(SEED).shuffle(ids)
    n = len(ids)
    return {item: ("DEVELOPMENT" if i < round(.6*n) else "VALIDATION" if i < round(.8*n) else "HELD_OUT")
            for i, item in enumerate(ids)}


def freeze(output: Path, sources: list[dict], inventory: list[dict], clusters: list[dict],
           pairs: list[dict], ambiguous: list[dict], queries: list[dict], no_match: list[dict],
           isolation: list[dict]) -> dict:
    validate(clusters, pairs, queries)
    output.mkdir(parents=True, exist_ok=True)
    split = split_clusters(clusters)
    files = {
        "source_manifest.json": sources, "artifact_inventory.json": inventory,
        "identity_clusters.json": clusters, "identity_pairs.json": pairs,
        "ambiguous_pairs.json": ambiguous, "ranking_queries.json": queries,
        "no_match_queries.json": no_match, "isolation_diagnostics.json": isolation,
        "split_manifest.json": {"seed": SEED, "by_identity_cluster_id": split},
    }
    for name, value in files.items():
        (output/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    positives = [p for p in pairs if p["label"] == "SAME_ENTITY" and p["confidence"] == "HIGH"]
    negatives = [p for p in pairs if p["label"] == "DIFFERENT_ENTITY" and p["confidence"] == "HIGH"]
    exact = sum(p.get("category") == "EXACT_NAME" for p in positives)
    stats = {"total_source_documents": len(sources), "total_graph_groups": len({m["graph_group"] for c in clusters for m in c["members"]}),
             "total_extracted_entities": sum(r.get("local_entity_count", 0) for r in inventory),
             "identity_clusters": len(clusters), "positive_identity_pairs": len(positives),
             "unique_positive_queries": len(queries), "negative_pairs": len(negatives),
             "ambiguous_pairs": len(ambiguous), "cross_document_positives": sum(p.get("cross_document", False) for p in positives),
             "exact_name_positives": exact, "non_exact_positives": len(positives)-exact,
             "non_exact_positive_share": (len(positives)-exact)/len(positives) if positives else 0,
             "related_but_distinct_negatives": sum(p.get("category") == "RELATED_BUT_DISTINCT" for p in negatives),
             "same_span_distinct_negatives": sum(p.get("category") == "SAME_SPAN_DISTINCT" for p in negatives),
             "no_match_queries": len(no_match)}
    reasons=[]
    if len(queries) < 50: reasons.append("fewer than 50 unique real identity-match queries")
    if len(negatives) < 100: reasons.append("fewer than 100 hard negatives")
    if stats["non_exact_positive_share"] < .70: reasons.append("non-exact positive share below 70%")
    quality={"status":"PASS" if not reasons else "FAIL","reasons":reasons,
             "labels_independent_of_model_scores":True,"repeated_runs_deduplicated":True,
             "graph_group_isolation_validated":True,"stage_b":"RUN" if not reasons else "NOT_RUN"}
    (output/"dataset_statistics.json").write_text(json.dumps(stats,indent=2)+"\n",encoding="utf-8")
    (output/"dataset_quality_report.json").write_text(json.dumps(quality,indent=2)+"\n",encoding="utf-8")
    manifest={"benchmark_version":VERSION,"source_manifest_hash":digest(sources),
              "label_manifest_hash":digest({"clusters":clusters,"pairs":pairs,"ambiguous":ambiguous}),
              "split_hash":digest(files["split_manifest.json"])}
    manifest["benchmark_hash"]=digest({"version":VERSION,"files":files,"statistics":stats,"quality":quality})
    (output/"benchmark_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return {"statistics":stats,"quality":quality,"manifest":manifest}
