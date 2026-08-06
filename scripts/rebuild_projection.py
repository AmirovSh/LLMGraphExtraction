"""Rebuild Neo4j from an authoritative successful fact_graph.json without model calls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import load_project_settings
from runtime.artifact_store import write_json
from runtime.final_graph_contract import validate_final_graph
from runtime.graph_view import GraphRun, validate_run_id, write_graph_view
from runtime.json_neo4j_parity import import_neo4j
from runtime.projection_manifest import load_projection_manifest, require_matching_projection

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config")
    args = parser.parse_args()
    run_id = validate_run_id(args.run_id)
    run_dir = args.outputs_dir / run_id
    graph = validate_final_graph(
        json.loads((run_dir / "fact_graph.json").read_text(encoding="utf-8"))
    ).model_dump(mode="json")
    settings = load_project_settings(args.config_dir)
    recorded_projection = load_projection_manifest(
        run_dir / "projection_manifest.json", graph_id=graph["graph_id"]
    )
    require_matching_projection(
        recorded=recorded_projection,
        configured=settings.graph_projection.type_names,
    )
    ownership = {
        "run_id": graph["run_id"],
        "source_sha256": graph["source"]["source_sha256"],
        "manifest_identity_hash": graph["manifest_identity_hash"],
    }
    neo4j, parity = import_neo4j(
        graph["graph_id"], graph["nodes"], graph["edges"], settings, ownership,
    )
    write_json(run_dir / "json_neo4j_edge_diff.json", parity)
    if parity.get("status") != "passed":
        raise RuntimeError("rebuilt Neo4j projection failed parity")
    selection = GraphRun(
        graph["run_id"], graph["graph_id"], run_dir, len(graph["nodes"]), len(graph["edges"]),
    )
    view = write_graph_view(run_dir, selection, settings)
    print(f"Graph ID: {selection.graph_id}")
    print(f"Neo4j nodes: {neo4j['nodes']}")
    print(f"Neo4j relationships: {neo4j['edges']}")
    print(f"Neo4j Browser: {view['browser_url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
