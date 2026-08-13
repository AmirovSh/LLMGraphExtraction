import json

from devtools.real_artifact_reranker_inventory import inventory_reports
from devtools.finalize_real_artifact_reranker_inventory import ARTIFACTS, finalize


def test_inventory_deduplicates_evidence_and_rejects_automatic_merges(tmp_path):
    for name in ("run_a", "run_b"):
        run = tmp_path / name
        run.mkdir()
        (run / "entity_resolution.json").write_text(json.dumps({"candidate_pairs": [
            {"left_local_id": "M1", "right_local_id": "M2", "decision": "MERGE",
             "explicit_equivalence_evidence": False},
            {"left_local_id": "M3", "right_local_id": "M4", "decision": "MERGE",
             "explicit_equivalence_evidence": True,
             "equivalence_relation_ids": ["D1:R1"],
             "equivalence_evidence_span_ids": ["D1:S1"]},
        ]}), encoding="utf-8")

    result = inventory_reports([tmp_path])

    assert result["unique_source_grounded_identity_pairs"] == 1
    assert result["maximum_bidirectional_identity_queries"] == 2
    assert result["dataset_gate"] == "INSUFFICIENT_DATA"
    assert len(result["unique_evidence"][0]["observed_in"]) == 2


def test_evidenceless_explicit_flag_is_not_ground_truth(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "resolution_report.json").write_text(json.dumps({"candidate_pairs": [{
        "left_local_id": "M1", "right_local_id": "M2",
        "explicit_equivalence_evidence": True,
    }]}), encoding="utf-8")

    result = inventory_reports([tmp_path])

    assert result["unique_source_grounded_identity_pairs"] == 0


def test_failed_gate_finalizer_marks_every_experiment_not_run(tmp_path):
    output = tmp_path / "audit"
    output.mkdir()
    inventory = output / "artifact_inventory.json"
    inventory.write_text(json.dumps({
        "dataset_gate": "INSUFFICIENT_DATA", "runs": [],
        "unique_source_grounded_identity_pairs": 1,
        "maximum_bidirectional_identity_queries": 2,
        "minimum_identity_queries": 50,
    }), encoding="utf-8")
    report = tmp_path / "report.md"

    finalize(inventory, report)

    assert "INSUFFICIENT_DATA" in report.read_text(encoding="utf-8")
    for name in ARTIFACTS:
        payload = json.loads((output / name).read_text(encoding="utf-8"))
        assert payload["status"] == "NOT_RUN"
        assert payload["reranker_calls"] == 0
