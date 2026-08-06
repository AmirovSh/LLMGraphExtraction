from __future__ import annotations

import json
from pathlib import Path

from devtools.project_conformance import (
    check_artifact_conformance, offline_test_files, run_offline_conformance,
)


def test_offline_conformance_orchestrates_real_contract_checks() -> None:
    result = run_offline_conformance()
    assert result["status"] == "passed"
    assert result["live_environment_checked"] is False
    assert result["checks"]["offline_contract_tests"]["status"] == "passed"
    assert result["checks"]["public_golden"]["status"] == "passed"
    assert "tests/test_project_conformance.py" not in offline_test_files()


def test_offline_test_files_are_loaded_from_project_configuration() -> None:
    assert offline_test_files() == (
        "tests/test_agent_skills.py",
        "tests/test_final_graph_contract.py",
        "tests/test_projection_schema_parity.py",
        "tests/test_graph_semantic_invariants.py",
        "tests/test_text_span_integrity.py",
        "tests/test_public_golden_contract.py",
    )


def test_artifact_conformance_fails_closed_when_contract_artifacts_are_missing(
    tmp_path: Path,
) -> None:
    result = check_artifact_conformance(tmp_path)
    assert result["status"] == "failed"
    assert "missing artifact: fact_graph.json" in result["errors"]


def test_conformance_cli_exposes_required_modes() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "devtools" / "check_project_conformance.py").read_text(
        encoding="utf-8"
    )
    for option in ("--offline", "--artifact-run", "--json-output"):
        assert option in source


def test_public_ci_runs_only_offline_checks() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    for job in (
        "public-install-and-offline-validation:",
        "agent-skills-validation:",
    ):
        assert job in workflow
    assert "os: [ubuntu-latest, windows-latest]" in workflow
    assert 'python -m pip install -c constraints.txt -e ".[test]"' in workflow
    assert "python -m pytest" in workflow
    assert "python -m devtools.check_project_conformance --offline" in workflow
    assert "python -m devtools.check_public_golden --offline" in workflow
    for module in (
        "scripts.run_production",
        "scripts.rebuild_projection",
        "scripts.open_graph",
        "scripts.open_graph",
    ):
        assert f"python -m {module} --help" in workflow
    assert "--live" not in workflow
    assert "NEO4J_PASSWORD" not in workflow
