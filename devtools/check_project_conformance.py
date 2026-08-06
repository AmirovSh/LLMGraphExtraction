"""Run offline or artifact-bound project and agent-skill conformance checks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.artifact_store import write_json
from devtools.project_conformance import (
    check_artifact_conformance, run_offline_conformance,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--artifact-run")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--git-diff-check", action="store_true")
    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    if args.offline:
        result = run_offline_conformance(
            include_git_diff_check=args.git_diff_check,
        )
    else:
        result = check_artifact_conformance(args.outputs_dir / args.artifact_run)
    if args.json_output:
        write_json(args.json_output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
