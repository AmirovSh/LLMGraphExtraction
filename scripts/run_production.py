"""Run the supported single-pass fact-extraction production pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import load_project_settings
from runtime.production_runner import run
from runtime.graph_view import (
    ViewerPreflightError,
    load_successful_run,
    open_browser,
    preflight_graph,
    validate_run_id,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", type=Path)
    output.add_argument("--run-id")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config")
    parser.add_argument("--candidate-threshold", type=float)
    parser.add_argument("--merge-threshold", type=float)
    parser.add_argument("--open-graph", action="store_true")
    args = parser.parse_args()
    output_dir = args.output or ROOT / "outputs" / validate_run_id(args.run_id)
    result = run(
        args.input, output_dir, args.config_dir, run_id=args.run_id,
        candidate_threshold=args.candidate_threshold, merge_threshold=args.merge_threshold,
    )
    if result:
        return result
    try:
        selection = load_successful_run(output_dir)
        view = json.loads((output_dir / "graph_view.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"Warning: graph view unavailable: {error}", file=sys.stderr)
        return 0
    print("PRODUCTION GRAPH BUILD COMPLETE")
    print()
    print(f"Run ID: {selection.run_id}")
    print(f"Graph ID: {selection.graph_id}")
    print(f"Units: {len(json.loads((output_dir / 'fact_graph.json').read_text(encoding='utf-8'))['source']['units'])}")
    print(f"Nodes: {selection.node_count}")
    print(f"Edges: {selection.relationship_count}")
    print(f"JSON graph: {(output_dir / 'fact_graph.json').resolve()}")
    print("Neo4j projection: passed")
    print("Parity: passed")
    print("Open graph:")
    print(f"  python -m scripts.open_graph --run-id {selection.run_id} --open")
    print("Neo4j Browser:")
    print(f"  {view['browser_url']}")
    if args.open_graph:
        try:
            preflight = preflight_graph(selection, load_project_settings(args.config_dir))
        except ViewerPreflightError as error:
            print(f"Warning: graph preflight failed; Browser was not opened: {error}", file=sys.stderr)
            return 0
        print(f"Preflight nodes: {preflight.nodes}")
        print(f"Preflight relationships: {preflight.relationships}")
        print(f"Preflight paths: {preflight.paths}")
        print("After Browser opens:")
        print("  1. Sign in if requested.")
        print("  2. Run :clear")
        print(f"  3. Run :use {preflight.database}")
        print("  4. Press Run or Ctrl+Enter.")
        opened, warning = open_browser(view["browser_url"])
        if not opened:
            print(f"Warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
