"""Build the independently curated Phase-3 benchmark before model scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from devtools.entity_resolution_reranker_phase3 import (
    Phase3Benchmark, Phase3Case, assert_split_isolation, fixed_phase3_split, freeze_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def _entity(name: str, kind: str, aliases: list[str] | None = None, group: str = "v3") -> dict:
    return {"canonical_name": name, "entity_type": kind, "aliases": aliases or [], "graph_group": group}


def build(definitions: dict) -> Phase3Benchmark:
    cases: list[Phase3Case] = []
    for section, category in (("abbreviation_positives", "abbreviation_expansion"),
                              ("short_full_positives", "short_full_name"),
                              ("surface_positives", "exact_case_punctuation")):
        for key, full, short, kind in definitions[section]:
            cases.append(Phase3Case(case_id=f"merge_{key}", identity_cluster_id=f"positive_{key}", category=category,
                entity_a=_entity(full, kind, [short]), entity_b=_entity(short, kind, [full]), expected="MERGE",
                label_reason="The reviewed short and full surfaces designate the same formal identity.",
                source_kind="new_manual_formal_identity_fixture", source_reference=f"phase3:{section}:{key}"))
    for key, acronym, correct, wrong, kind in definitions["acronym_negatives"]:
        cases.append(Phase3Case(case_id=f"split_acronym_{key}", identity_cluster_id=f"negative_acronym_{key}",
            category="abbreviation_hard_negative", entity_a=_entity(acronym, kind, [correct]),
            entity_b=_entity(wrong, kind), expected="SPLIT",
            label_reason="The same short surface has a different reviewed expansion.",
            source_kind="new_counterfactual_formal_identity_fixture", source_reference=f"phase3:acronym:{key}"))
    for key, name, kind, left_context, right_context in definitions["same_surface_negatives"]:
        cases.append(Phase3Case(case_id=f"split_surface_{key}", identity_cluster_id=f"negative_surface_{key}",
            category="same_surface_negative", entity_a=_entity(name, kind, [left_context]),
            entity_b=_entity(name, kind, [right_context]), expected="SPLIT",
            label_reason="Identical labels belong to explicitly distinct reviewed owners or contexts.",
            source_kind="new_manual_same_surface_fixture", source_reference=f"phase3:same_surface:{key}",
            deterministic_block="EXPLICIT_DISTINCT"))
    for section, category, block in (("version_negatives", "version", "VERSION_CONFLICT"),
                                     ("opposition_negatives", "opposition", "OPPOSITION_CONFLICT")):
        for key, left, right, kind in definitions[section]:
            cases.append(Phase3Case(case_id=f"split_{key}", identity_cluster_id=f"negative_{key}", category=category,
                entity_a=_entity(left, kind), entity_b=_entity(right, kind), expected="SPLIT",
                label_reason=f"The pair has an explicit reviewed {category} distinction.",
                source_kind="new_manual_formal_identity_fixture", source_reference=f"phase3:{section}:{key}",
                deterministic_block=block))
    for key, left, right, category in definitions["structural_negatives"]:
        cases.append(Phase3Case(case_id=f"split_structural_{key}", identity_cluster_id=f"negative_structural_{key}",
            category="structural_related_but_distinct" if category not in {"policy_result", "process_record"} else category,
            entity_a=_entity(left, "CONCEPT"), entity_b=_entity(right, "CONCEPT"), expected="SPLIT",
            label_reason=f"Reviewed identities are related but distinct ({category}).",
            source_kind="new_synthetic_formal_document_fixture", source_reference=f"phase3:structural:{key}"))
    # Diagnostic counterfactuals are paired with abbreviation positives but never tuned/evaluated as primary data.
    for index, (key, full, short, kind) in enumerate(definitions["abbreviation_positives"][:20]):
        wrong = definitions["acronym_negatives"][index][3]
        cases.append(Phase3Case(case_id=f"stress_{key}", identity_cluster_id=f"stress_{key}",
            category="counterfactual_abbreviation", entity_a=_entity(short, kind), entity_b=_entity(wrong, kind),
            expected="SPLIT", label_reason="Plausible but explicitly incorrect expansion for abbreviation stress testing.",
            source_kind="counterfactual_stress_fixture", source_reference=f"phase3:stress:{key}", diagnostic_only=True))
    return Phase3Benchmark(benchmark_version="3.1", cases=cases)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "release_audit/reranker_phase3")
    args = parser.parse_args()
    definitions = json.loads((ROOT / "devtools/fixtures/entity_resolution_reranker_phase3_definitions.json").read_text(encoding="utf-8"))
    benchmark = build(definitions)
    split = fixed_phase3_split(benchmark.cases)
    assert_split_isolation(benchmark.cases, split)
    manifest = freeze_manifest(benchmark, split)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "benchmark_v3.json").write_text(json.dumps(benchmark.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    (args.output / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    primary = [case for case in benchmark.cases if not case.diagnostic_only]
    print(json.dumps({"fresh_eligible": len(primary), "merge": sum(c.expected == "MERGE" for c in primary),
                      "split": sum(c.expected == "SPLIT" for c in primary), "manifest": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
