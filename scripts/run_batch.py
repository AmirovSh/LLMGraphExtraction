"""CLI for manifest-driven multi-document graph batches."""
from __future__ import annotations

import argparse
from pathlib import Path

from runtime.batch_runner import run_batch


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a grouped multi-document fact-graph batch")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    args = parser.parse_args()
    return run_batch(args.manifest, args.config_dir, output_override=args.output)


if __name__ == "__main__":
    raise SystemExit(main())
