from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from devtools.public_golden import (
    DEFAULT_MANIFEST, load_manifest,
    prompt_contract_metadata, run_offline_public_golden,
)


def test_offline_public_golden_is_clean_no_resume_and_does_not_mutate_manifest() -> None:
    before = DEFAULT_MANIFEST.read_bytes()
    result = run_offline_public_golden()
    assert result["status"] == "passed"
    assert result["mode"] == "offline"
    assert result["live_model_acceptance"] is False
    assert result["clean_namespace"] is True
    assert result["resume_used"] is False
    assert result["structured_units_consumed"] == 42
    assert DEFAULT_MANIFEST.read_bytes() == before


def test_public_golden_binds_prompt_identity_and_hash() -> None:
    metadata = prompt_contract_metadata()
    manifest = load_manifest()
    assert metadata["actual_prompt_id"] == metadata["expected_prompt_id"]
    assert metadata["actual_prompt_hash"] == manifest["prompt_hash"]


def test_semantic_acceptance_has_no_external_metadata_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "devtools.public_golden.sha256_json",
        lambda _value: "f" * 64,
    )
    result = run_offline_public_golden()
    manifest = load_manifest()
    assert result["status"] == "passed"
    assert result["observed_metrics"]["graph_hash"] == "f" * 64
    assert result["failed_assertions"] == []
    assert result["contract_version"] == manifest["contract_version"]
    serialized = json.dumps(result, sort_keys=True)
    for forbidden in (
        "recommen" + "dation", "suggested" + "_changes", "review" + "er",
        "approval" + "_status", "reviewed" + "_graph_hash",
    ):
        assert forbidden not in serialized


def test_new_graph_hash_is_rejected_only_when_semantics_differ(
    tmp_path: Path,
) -> None:
    manifest = load_manifest()
    manifest["required_entities"].append("entity absent from candidate")
    path = tmp_path / "semantic_mismatch.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = run_offline_public_golden(path)
    assert result["status"] == "failed"
    assert result["observed_metrics"]["projection_parity_status"] == "passed"
    assert any(
        assertion == "missing required entity: entity absent from candidate"
        for assertion in result["failed_assertions"]
    )


def test_projection_parity_and_semantic_assertions_are_independent(
    tmp_path: Path,
) -> None:
    manifest = load_manifest()
    manifest["projection_contract"]["parity_status"] = "failed"
    path = tmp_path / "parity_mismatch.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = run_offline_public_golden(path)
    assert result["status"] == "failed"
    assert result["observed_metrics"]["projection_parity_status"] == "passed"
    assert result["failed_assertions"] == [
        "projection parity status differs from golden contract"
    ]


def test_normal_golden_validation_cannot_update_manifest() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "devtools" / "check_public_golden.py").read_text(
        encoding="utf-8"
    )
    assert "--write" not in source
    assert "--approve" not in source
    removed_workflow = "review" + "_golden_update.py"
    assert not (root / "scripts" / removed_workflow).exists()


def test_public_manifest_contains_automatic_endpoint_and_object_assertions() -> None:
    manifest = load_manifest()
    assert {
        "source": "Audit Portal",
        "raw_relation": "displays",
        "target": "evidence",
        "negated": False,
    } in manifest["required_relations"]
    assert {
        "source": "Harbor Operations Hub",
        "raw_relation": "coordinates",
        "target": "North Pier",
    } in manifest["forbidden_relations"]
    assert {
        "source": "Audit Portal",
        "raw_relation": "displays",
        "target": "Operations Ledger",
    } in manifest["forbidden_relations"]


def test_live_run_id_mode_validates_existing_artifact_without_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import devtools.check_public_golden as cli

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "check_artifact_run",
        lambda run_dir, manifest: captured.update(
            run_dir=run_dir, manifest=manifest,
        ) or {
            "status": "passed",
            "failed_assertions": [],
            "observed_metrics": {},
            "contract_version": "1",
        },
    )
    monkeypatch.setattr(
        cli, "run", lambda *_args, **_kwargs: pytest.fail("model run invoked"),
    )
    monkeypatch.setattr(
        sys, "argv",
        ["check_public_golden", "--live", "--run-id", "existing_run"],
    )
    assert cli.main() == 0
    assert Path(captured["run_dir"]).name == "existing_run"


def test_report_only_policy_does_not_turn_semantic_miss_into_exit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import devtools.check_public_golden as cli

    monkeypatch.setattr(
        cli,
        "check_artifact_run",
        lambda *_args: {
            "status": "failed",
            "failed_assertions": ["missing required relation: example"],
            "assertion_counts": {"passed": 8, "total": 9},
            "observed_metrics": {},
            "contract_version": "2",
        },
    )
    report = tmp_path / "development_golden_report.json"
    monkeypatch.setattr(
        sys, "argv", [
            "check_public_golden", "--live", "--run-id", "existing_run",
            "--policy", "report-only", "--json-output", str(report),
        ],
    )
    assert cli.main() == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["blocking"] is False
    assert payload["development_status"] == "partial"
