"""Run bounded Phase-3R context diagnostics after the primary fixed-pair test."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from config.settings import environment_value, load_project_settings
from devtools.contextual_entity_profiles import render_profile
from devtools.entity_resolution_reranker import DirectionalPair, RerankerClient, RerankerConfig, join_directional_results
from devtools.run_contextual_reranker import ROOT, benchmark, load_local_env

OUT = ROOT / "release_audit/context_reranker"


def strip_section(text: str, heading: str, next_heading: str | None) -> str:
    start = text.find(heading)
    if start < 0:
        return text
    end = text.find(next_heading, start) if next_heading else len(text)
    if end < 0:
        end = len(text)
    return (text[:start] + text[end:]).strip()


def score(client: RerankerClient, profiles: dict[str, tuple[str, str]], *, batch_size: int = 16) -> dict:
    pairs = []
    for case_id, (left, right) in profiles.items():
        pairs.extend((DirectionalPair(case_id, left, right, "ab"), DirectionalPair(case_id, right, left, "ba")))
    return join_directional_results(client.score_many(pairs, batch_size=batch_size))


def main() -> int:
    load_local_env(ROOT / ".env.runtime"); settings = load_project_settings(ROOT / "config"); remote = settings.runtime.remote
    client = RerankerClient(RerankerConfig(environment_value(remote.embedding_base_url_env),
        environment_value(remote.embedding_api_key_env), timeout_seconds=remote.embedding_timeout_seconds, trust_env=remote.trust_env))
    cases = [case for index, case in enumerate(benchmark()) if index % 5 == 4]
    base: dict[str, dict[str, tuple[str, str]]] = {name: {} for name in ("R2", "R3", "R4")}
    variants: dict[str, dict[str, tuple[str, str]]] = {name: {} for name in
        ("R4", "R4_without_evidence", "R4_without_relations", "R4_without_aliases", "R4_without_type", "R4_context_swapped")}
    for case in cases:
        cid = case["case_id"]; a = case["entity_a"]; b = case["entity_b"]
        for rep in base:
            base[rep][cid] = (render_profile(a, rep)["text"], render_profile(b, rep)["text"])
        left, right = base["R4"][cid]; variants["R4"][cid] = (left, right)
        variants["R4_without_evidence"][cid] = (strip_section(left, "evidence:", "observed_local_relations:"), strip_section(right, "evidence:", "observed_local_relations:"))
        variants["R4_without_relations"][cid] = (strip_section(left, "observed_local_relations:", None), strip_section(right, "observed_local_relations:", None))
        variants["R4_without_aliases"][cid] = (strip_section(left, "aliases:", "evidence:"), strip_section(right, "aliases:", "evidence:"))
        variants["R4_without_type"][cid] = ("\n".join(x for x in left.splitlines() if not x.startswith("type:")), "\n".join(x for x in right.splitlines() if not x.startswith("type:")))
        # Names/types/aliases remain on their entity; only evidence+incidents are swapped.
        a_head = strip_section(strip_section(left, "evidence:", "observed_local_relations:"), "observed_local_relations:", None)
        b_head = strip_section(strip_section(right, "evidence:", "observed_local_relations:"), "observed_local_relations:", None)
        a_context = left[left.find("evidence:"):]; b_context = right[right.find("evidence:"):]
        variants["R4_context_swapped"][cid] = (a_head + "\n" + b_context, b_head + "\n" + a_context)
    ablation_scores = {name: score(client, profiles) for name, profiles in variants.items()}
    (OUT / "context_ablation.json").write_text(json.dumps({k: v for k, v in ablation_scores.items() if k != "R4_context_swapped"}, indent=2) + "\n")
    swap = {cid: {"correct_mean": ablation_scores["R4"][cid]["mean"], "swapped_mean": ablation_scores["R4_context_swapped"][cid]["mean"],
                  "absolute_change": abs(ablation_scores["R4"][cid]["mean"] - ablation_scores["R4_context_swapped"][cid]["mean"])} for cid in variants["R4"]}
    (OUT / "context_swap_diagnostic.json").write_text(json.dumps({"pairs": swap, "mean_absolute_change": statistics.mean(x["absolute_change"] for x in swap.values())}, indent=2) + "\n")
    repeats = {rep: [score(client, profiles) for _ in range(5)] for rep, profiles in base.items()}
    stability = {}
    for rep, runs in repeats.items():
        rows=[]; drift=0
        for cid in base[rep]:
            values=[run[cid]["mean"] for run in runs]; changed=max(values)-min(values)
            rows.append({"case_id":cid,"mean":statistics.mean(values),"stddev":statistics.pstdev(values),"range":changed})
            drift += changed > 1e-5
        stability[rep]={"runs":5,"pairs":len(rows),"mean_range":statistics.mean(x["range"] for x in rows),"pairs_with_range_over_1e-5":drift,"rows":rows}
    (OUT / "stability.json").write_text(json.dumps(stability, indent=2) + "\n")
    print(json.dumps({"context_swap_mean_absolute_change": statistics.mean(x["absolute_change"] for x in swap.values()), "stability": {k:{x:v[x] for x in ("pairs","mean_range","pairs_with_range_over_1e-5")} for k,v in stability.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
