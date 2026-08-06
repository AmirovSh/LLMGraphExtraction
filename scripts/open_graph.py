"""Print or open a Neo4j Browser view of a completed fact graph."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import load_project_settings
from runtime.graph_view import (
    ViewerPreflightError,
    graph_view_payload,
    open_browser,
    preflight_graph,
    select_run,
    validate_limit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selectors = parser.add_mutually_exclusive_group()
    selectors.add_argument("--latest", action="store_true")
    selectors.add_argument("--run-id")
    selectors.add_argument("--graph-id")
    parser.add_argument("--limit", type=int, default=500)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--open", action="store_true")
    action.add_argument("--print-only", action="store_true")
    parser.add_argument("--show-query", action="store_true")
    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs", help=argparse.SUPPRESS)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config", help=argparse.SUPPRESS)
    args = parser.parse_args()
    limit = validate_limit(args.limit)
    selection = select_run(args.outputs_dir, run_id=args.run_id, graph_id=args.graph_id)
    settings = load_project_settings(args.config_dir)
    payload = graph_view_payload(selection, settings, limit=limit)
    try:
        preflight = preflight_graph(selection, settings, limit=limit)
    except ViewerPreflightError as error:
        print(f"Error: {error}", file=sys.stderr)
        if error.diagnostics:
            print("Preflight diagnostics:", file=sys.stderr)
            for name, value in error.diagnostics.items():
                print(f"  {name}: {value}", file=sys.stderr)
        return 2
    print(f"Selected run: {selection.run_id or '(graph-id selector)'}")
    print(f"Graph ID: {selection.graph_id}")
    print(f"Neo4j database: {preflight.database}")
    print(f"Preflight nodes: {preflight.nodes}")
    print(f"Preflight relationships: {preflight.relationships}")
    print(f"Preflight paths: {preflight.paths}")
    print("Relationship caption: raw_relation")
    print("Browser stylesheet: config/neo4j_browser.grass")
    print("Browser style is not installed in this Browser session.")
    print("Run:")
    print("  :style")
    print("Then upload:")
    print("  config/neo4j_browser.grass")
    if args.show_query:
        print("Cypher:")
        print(payload["query"])
    print("Browser URL:")
    print(payload["browser_url"])
    print("After Browser opens:")
    print("  1. Sign in if requested.")
    print("  2. Run :clear")
    print(f"  3. Run :use {preflight.database}")
    print("  4. Press Run or Ctrl+Enter.")
    if args.open:
        opened, warning = open_browser(payload["browser_url"])
        if not opened:
            print(f"Warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
