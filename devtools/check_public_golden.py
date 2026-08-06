"""Run immutable offline or clean live public-sample golden validation."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.artifact_store import write_json
from runtime.production_runner import run
from runtime.graph_view import validate_run_id
from devtools.public_golden import (
    DEFAULT_MANIFEST, check_artifact_run, load_manifest, run_offline_public_golden,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config")
    parser.add_argument(
        "--run-id",
        help="Validate an existing completed run in live mode without model calls.",
    )
    parser.add_argument(
        "--policy", choices=("blocking", "report-only"), default="blocking",
        help="Treat development semantic assertions as a gate or report only.",
    )
    parser.add_argument(
        "--non-blocking", action="store_true",
        help="Alias for --policy report-only.",
    )
    args = parser.parse_args()
    if args.offline:
        if args.run_id:
            parser.error("--run-id is available only with --live")
        result = run_offline_public_golden(args.manifest)
    else:
        manifest = load_manifest(args.manifest)
        if args.run_id:
            run_dir = ROOT / "outputs" / validate_run_id(args.run_id)
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_dir = ROOT / "outputs" / f"public_golden_live_{timestamp}"
            run(
                ROOT / manifest["fixture"], run_dir, args.config_dir,
                run_id=f"public_golden_{timestamp}",
            )
        result = check_artifact_run(run_dir, args.manifest)
        result.update({
            "mode": "live",
            "live_model_acceptance": True,
            "clean_namespace": not args.run_id,
            "resume_used": False,
            "existing_run_validated": bool(args.run_id),
        })
    report_only = args.non_blocking or args.policy == "report-only"
    result["blocking"] = not report_only
    if report_only:
        failures = result.get("failed_assertions") or []
        total = int(result.get("assertion_counts", {}).get("total", 0))
        passed = int(result.get("assertion_counts", {}).get("passed", 0))
        result["development_status"] = (
            "pass" if not failures else "partial" if passed else "fail"
        )
        if args.live and args.run_id and not args.json_output:
            args.json_output = (
                ROOT / "outputs" / validate_run_id(args.run_id)
                / "development_golden_report.json"
            )
    if args.json_output:
        write_json(args.json_output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if report_only or result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
