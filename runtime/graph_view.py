"""Secret-free Neo4j Browser links and successful-run discovery."""
from __future__ import annotations

import json
import os
import re
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode, urlsplit, urlunsplit

from neo4j import GraphDatabase

from config.projection_identifiers import DEFAULT_PROJECTION_TYPES, ProjectionTypeNames
from config.settings import ProjectSettings, environment_value
from runtime.artifact_store import write_json
from runtime.projection_manifest import load_projection_manifest

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_GRAPH_ID = re.compile(r"^fact_extraction_[A-Za-z0-9][A-Za-z0-9._-]*$")
_PARITY_FAILURE_FIELDS = (
    "json_only_nodes", "neo4j_only_nodes", "json_only_edges", "neo4j_only_edges",
    "source_target_mismatches", "edges_without_graph_id", "duplicate_edges", "import_errors",
)


@dataclass(frozen=True)
class GraphRun:
    run_id: str
    graph_id: str
    run_dir: Path | None
    node_count: int | None = None
    relationship_count: int | None = None
    completed_at_utc: str | None = None
    projection: ProjectionTypeNames | None = None


@dataclass(frozen=True)
class GraphPreflight:
    database: str
    nodes: int
    relationships: int
    paths: int
    filter_paths: dict[str, int]


class ViewerPreflightError(RuntimeError):
    def __init__(self, message: str, *, diagnostics: dict[str, int] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def validate_run_id(value: str) -> str:
    if not _RUN_ID.fullmatch(value):
        raise ValueError("run_id may contain only letters, numbers, dot, underscore, and hyphen")
    return value


def validate_graph_id(value: str) -> str:
    if not _GRAPH_ID.fullmatch(value):
        raise ValueError("graph_id must start with fact_extraction_ and contain only safe identifier characters")
    return value


def validate_limit(value: int) -> int:
    if not 1 <= value <= 5000:
        raise ValueError("limit must be between 1 and 5000")
    return value


def graph_query(
    graph_id: str, limit: int = 500,
    projection: ProjectionTypeNames = DEFAULT_PROJECTION_TYPES,
) -> str:
    graph_id = validate_graph_id(graph_id)
    limit = validate_limit(limit)
    literal = json.dumps(graph_id)
    entity_label = projection.entity_label.cypher
    relation_type = projection.relation_type.cypher
    return (
        f"MATCH p=(source:{entity_label})-[relation:{relation_type}]->(target:{entity_label})\n"
        f"WHERE source.graph_id = {literal}\n"
        f"  AND relation.graph_id = {literal}\n"
        f"  AND target.graph_id = {literal}\n"
        "RETURN p\n"
        f"LIMIT {limit}"
    )


def _credential_free_uri(uri: str) -> str:
    parsed = urlsplit(uri)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError("Neo4j connection URI must include a scheme and host")
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    netloc = f"{host}:{parsed.port}" if parsed.port else host
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


def browser_url(*, base_url: str, connection_uri: str, database: str, query: str) -> str:
    if not base_url:
        raise ValueError("Neo4j Browser base URL is not configured")
    return f"{base_url.rstrip('?')}?{urlencode({'dbms': _credential_free_uri(connection_uri), 'db': database, 'cmd': 'edit', 'arg': query})}"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_successful_run(run_dir: Path) -> GraphRun:
    graph_path = run_dir / "fact_graph.json"
    required = (graph_path, run_dir / "run_report.json", run_dir / "validation_report.json",
                run_dir / "json_neo4j_edge_diff.json", run_dir / "projection_manifest.json")
    if not all(path.is_file() for path in required):
        raise ValueError(f"run is incomplete: {run_dir.name}")
    graph = _read_json(graph_path)
    report = _read_json(run_dir / "run_report.json")
    validation = _read_json(run_dir / "validation_report.json")
    parity = _read_json(run_dir / "json_neo4j_edge_diff.json")
    graph_id = validate_graph_id(str(graph.get("graph_id", "")))
    projection = load_projection_manifest(
        run_dir / "projection_manifest.json", graph_id=graph_id
    )
    run_id = validate_run_id(str(graph.get("run_id", "")))
    if run_id != run_dir.name or report.get("run_id") != run_id or report.get("graph_id") != graph_id:
        raise ValueError(f"run identity mismatch: {run_dir.name}")
    if not validation.get("passed") or not validation.get("json_neo4j_parity"):
        raise ValueError(f"run validation failed: {run_id}")
    if any(parity.get(field) for field in _PARITY_FAILURE_FIELDS):
        raise ValueError(f"run parity failed: {run_id}")
    neo4j = report.get("neo4j") or {}
    if (neo4j.get("graph_id"), neo4j.get("nodes"), neo4j.get("edges")) != (
        graph_id, len(graph.get("nodes") or []), len(graph.get("edges") or []),
    ):
        raise ValueError(f"run projection is incomplete: {run_id}")
    view_path = run_dir / "graph_view.json"
    view = _read_json(view_path) if view_path.is_file() else {}
    return GraphRun(run_id, graph_id, run_dir, len(graph["nodes"]), len(graph["edges"]),
                    view.get("completed_at_utc") or report.get("completed_at_utc"), projection)


def select_run(outputs_dir: Path, *, run_id: str | None = None, graph_id: str | None = None) -> GraphRun:
    if run_id and graph_id:
        raise ValueError("choose only one of run_id or graph_id")
    if graph_id:
        return GraphRun("", validate_graph_id(graph_id), None)
    if run_id:
        return load_successful_run(outputs_dir / validate_run_id(run_id))
    candidates: list[GraphRun] = []
    if outputs_dir.is_dir():
        for child in outputs_dir.iterdir():
            if not child.is_dir():
                continue
            try:
                candidate = load_successful_run(child)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if candidate.completed_at_utc:
                candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError("no completed successful run with completion metadata was found")
    return max(candidates, key=lambda item: (item.completed_at_utc or "", item.run_id))


def graph_view_payload(
    selection: GraphRun, settings: ProjectSettings, *, limit: int = 500,
    completed_at_utc: str | None = None,
) -> dict[str, Any]:
    projection = selection.projection or settings.graph_projection.type_names
    query = graph_query(selection.graph_id, limit, projection)
    connection_uri = environment_value(settings.neo4j.uri_env)
    database = os.environ.get(settings.neo4j.database_env, settings.neo4j.default_database)
    return {
        "run_id": selection.run_id,
        "graph_id": selection.graph_id,
        "browser_url": browser_url(
            base_url=settings.neo4j.browser_base_url,
            connection_uri=connection_uri,
            database=database,
            query=query,
        ),
        "query": query,
        "database": database,
        "node_count": selection.node_count,
        "relationship_count": selection.relationship_count,
        "projection": projection.as_dict(),
        "completed_at_utc": completed_at_utc or datetime.now(timezone.utc).isoformat(),
    }


def preflight_graph(
    selection: GraphRun, settings: ProjectSettings, *, limit: int = 500,
    driver_factory: Callable[..., Any] = GraphDatabase.driver,
) -> GraphPreflight:
    graph_id = validate_graph_id(selection.graph_id)
    projection = selection.projection or settings.graph_projection.type_names
    entity_label = projection.entity_label.cypher
    relation_type = projection.relation_type.cypher
    query = graph_query(graph_id, limit, projection)
    database = os.environ.get(settings.neo4j.database_env, settings.neo4j.default_database)
    if database.casefold() == "system":
        raise ViewerPreflightError("refusing to query the system database")
    uri = environment_value(settings.neo4j.uri_env)
    username = environment_value(settings.neo4j.username_env)
    password = environment_value(settings.neo4j.password_env)
    filters = {
        "unfiltered": f"MATCH p=(:{entity_label})-[:{relation_type}]->(:{entity_label}) RETURN count(p) AS paths",
        "source": f"MATCH p=(source:{entity_label})-[relation:{relation_type}]->(target:{entity_label}) WHERE source.graph_id = $graph_id RETURN count(p) AS paths",
        "relationship": f"MATCH p=(source:{entity_label})-[relation:{relation_type}]->(target:{entity_label}) WHERE relation.graph_id = $graph_id RETURN count(p) AS paths",
        "target": f"MATCH p=(source:{entity_label})-[relation:{relation_type}]->(target:{entity_label}) WHERE target.graph_id = $graph_id RETURN count(p) AS paths",
        "all": f"MATCH p=(source:{entity_label})-[relation:{relation_type}]->(target:{entity_label}) WHERE source.graph_id = $graph_id AND relation.graph_id = $graph_id AND target.graph_id = $graph_id RETURN count(p) AS paths",
    }
    try:
        with driver_factory(uri, auth=(username, password)) as driver:
            with driver.session(database=database) as session:
                nodes = session.run(
                    f"MATCH (n:{entity_label} {{graph_id:$graph_id}}) RETURN count(n) AS count",
                    graph_id=graph_id,
                ).single()["count"]
                relationships = session.run(
                    f"MATCH ()-[r:{relation_type} {{graph_id:$graph_id}}]->() RETURN count(r) AS count",
                    graph_id=graph_id,
                ).single()["count"]
                filter_paths = {
                    name: session.run(statement, graph_id=graph_id).single()["paths"]
                    for name, statement in filters.items()
                }
                paths = len(list(session.run(query)))
    except ViewerPreflightError:
        raise
    except Exception as error:
        raise ViewerPreflightError(
            f"Neo4j preflight failed for database {database}: {type(error).__name__}: {error}"
        ) from error
    diagnostics = {"nodes": nodes, "relationships": relationships, "paths": paths, **filter_paths}
    if nodes <= 0:
        raise ViewerPreflightError(f"graph_id does not exist in database {database}", diagnostics=diagnostics)
    if relationships <= 0:
        raise ViewerPreflightError(f"graph_id has no relationships in database {database}", diagnostics=diagnostics)
    if paths <= 0:
        raise ViewerPreflightError("viewer query returned zero paths", diagnostics=diagnostics)
    return GraphPreflight(database, nodes, relationships, paths, filter_paths)


def write_graph_view(run_dir: Path, selection: GraphRun, settings: ProjectSettings, *, limit: int = 500) -> dict[str, Any]:
    target = run_dir / "graph_view.json"
    previous = _read_json(target) if target.is_file() else {}
    payload = graph_view_payload(
        selection, settings, limit=limit, completed_at_utc=previous.get("completed_at_utc"),
    )
    write_json(target, payload)
    return payload


def open_browser(url: str, *, opener: Callable[[str], Any] = webbrowser.open) -> tuple[bool, str | None]:
    try:
        opened = bool(opener(url))
        return opened, None if opened else "system browser did not accept the URL"
    except Exception as error:  # browser integration must not fail a completed pipeline
        return False, f"{type(error).__name__}: {error}"
