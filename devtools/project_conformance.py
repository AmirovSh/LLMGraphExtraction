"""Project conformance orchestration over existing validators and test contracts."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from runtime.final_graph_contract import validate_final_graph
from devtools.public_golden import check_artifact_run, run_offline_public_golden

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ID_PATTERN = re.compile(r"\bM[0-9]{3}\b")


def offline_test_files() -> tuple[str, ...]:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)
    configured = project["tool"]["semantic-fact-graph"]["conformance"][
        "offline-test-files"
    ]
    if (
        not isinstance(configured, list)
        or not configured
        or any(not isinstance(path, str) or not path for path in configured)
        or len(configured) != len(set(configured))
    ):
        raise ValueError("offline-test-files must be a non-empty list of unique paths")
    for configured_path in configured:
        path = Path(configured_path)
        if path.is_absolute() or ".." in path.parts or not (ROOT / path).is_file():
            raise ValueError(f"invalid offline test path: {configured_path!r}")
    return tuple(configured)


def _run_pytest() -> dict[str, Any]:
    process = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *offline_test_files()],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output_lines = (process.stdout + process.stderr).strip().splitlines()
    return {
        "status": "passed" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "summary": output_lines[-1] if output_lines else "",
    }


def _git_diff_check() -> dict[str, Any]:
    process = subprocess.run(
        ["git", "diff", "--check"], cwd=ROOT, text=True, capture_output=True,
        check=False,
    )
    return {
        "status": "passed" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "output": (process.stdout + process.stderr).strip(),
    }


def run_offline_conformance(*, include_git_diff_check: bool = False) -> dict[str, Any]:
    tests = _run_pytest()
    golden = run_offline_public_golden()
    checks: dict[str, Any] = {
        "offline_contract_tests": tests,
        "public_golden": golden,
    }
    if include_git_diff_check:
        checks["git_diff_check"] = _git_diff_check()
    status = "passed" if all(
        check.get("status") == "passed" for check in checks.values()
    ) else "failed"
    return {
        "status": status,
        "mode": "offline",
        "live_environment_checked": False,
        "checks": checks,
    }


def _contains_local_id(value: Any) -> bool:
    if isinstance(value, str):
        return bool(LOCAL_ID_PATTERN.search(value))
    if isinstance(value, dict):
        return any(_contains_local_id(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_local_id(item) for item in value)
    return False


def check_artifact_conformance(run_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    required = (
        "fact_graph.json", "projection_manifest.json", "model_call_budget.json", "json_neo4j_edge_diff.json",
        "validation_report.json", "run_report.json", "prompt_manifest.json",
        "completion_status.json",
    )
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        return {
            "status": "failed", "mode": "artifact",
            "errors": [f"missing artifact: {name}" for name in missing],
        }
    try:
        graph = validate_final_graph(
            json.loads((run_dir / "fact_graph.json").read_text(encoding="utf-8"))
        ).model_dump(mode="json")
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        return {
            "status": "failed", "mode": "artifact",
            "errors": [f"final graph contract failed: {type(error).__name__}: {error}"],
        }
    budget = json.loads((run_dir / "model_call_budget.json").read_text(encoding="utf-8"))
    parity = json.loads(
        (run_dir / "json_neo4j_edge_diff.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (run_dir / "validation_report.json").read_text(encoding="utf-8")
    )
    prompt = json.loads((run_dir / "prompt_manifest.json").read_text(encoding="utf-8"))
    completion = json.loads(
        (run_dir / "completion_status.json").read_text(encoding="utf-8")
    )
    totals = budget["totals"]
    unit_count = len(graph["source"]["units"])
    if totals["primary_successful_calls"] != unit_count:
        errors.append("primary successful calls differ from unit count")
    if totals["post_extraction_llm_calls"] != 0:
        errors.append("post-extraction semantic calls are nonzero")
    if parity.get("status") != "passed":
        errors.append("JSON/Neo4j data/schema/namespace parity failed")
    if not validation.get("passed"):
        errors.append("validation report is not passed")
    if prompt["prompt_id"] != graph["prompt_id"] or prompt["prompt_hash"] != graph["prompt_hash"]:
        errors.append("prompt manifest differs from final graph")
    if len({node["entity_id"] for node in graph["nodes"]}) != len(graph["nodes"]):
        errors.append("duplicate node IDs")
    if len({edge["edge_id"] for edge in graph["edges"]}) != len(graph["edges"]):
        errors.append("duplicate edge IDs")
    if any(not edge["raw_relation"].strip() for edge in graph["edges"]):
        errors.append("blank raw_relation")
    if any(_contains_local_id(edge["qualifiers"]) for edge in graph["edges"]):
        errors.append("unresolved local qualifier ID")
    expected_completion = {
        "run_status": "completed",
        "extraction_status": "passed",
        "schema_status": "passed",
        "evidence_status": "passed",
        "final_graph_status": "passed",
        "projection_status": "passed",
        "parity_status": "passed",
    }
    if completion != expected_completion:
        errors.append("production completion status contract failed")
    golden_result = check_artifact_run(run_dir)
    return {
        "status": "passed" if not errors else "failed",
        "mode": "artifact",
        "run_dir": str(run_dir),
        "errors": errors,
        "graph": {
            "graph_id": graph["graph_id"],
            "nodes": len(graph["nodes"]),
            "edges": len(graph["edges"]),
            "unit_count": unit_count,
        },
        "call_accounting": totals,
        "parity_status": parity.get("status"),
        "development_semantic_golden": {
            **golden_result,
            "blocking": False,
        },
    }
