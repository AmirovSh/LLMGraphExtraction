from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from config.settings import load_project_settings
from runtime.graph_view import (
    GraphPreflight,
    GraphRun,
    ViewerPreflightError,
    browser_url,
    graph_query,
    graph_view_payload,
    open_browser,
    preflight_graph,
    select_run,
    validate_limit,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeResult:
    def __init__(self, rows: list[dict]): self.rows = rows
    def single(self): return self.rows[0]
    def __iter__(self): return iter(self.rows)


class FakeSession:
    def __init__(self, *, database: str, nodes: int, relationships: int, paths: int, expected_database: str):
        if database != expected_database:
            raise RuntimeError("database not found")
        self.nodes = nodes; self.relationships = relationships; self.paths = paths
    def __enter__(self): return self
    def __exit__(self, *_args): return None
    def run(self, query: str, **_params):
        if "RETURN count(n) AS count" in query:
            return FakeResult([{"count": self.nodes}])
        if "RETURN count(r) AS count" in query:
            return FakeResult([{"count": self.relationships}])
        if "RETURN count(p) AS paths" in query:
            return FakeResult([{"paths": self.paths}])
        return FakeResult([{"p": object()} for _ in range(self.paths)])


class FakeDriver:
    def __init__(self, *, nodes: int, relationships: int, paths: int, expected_database: str):
        self.nodes = nodes; self.relationships = relationships; self.paths = paths
        self.expected_database = expected_database
    def __enter__(self): return self
    def __exit__(self, *_args): return None
    def session(self, *, database: str):
        return FakeSession(
            database=database, nodes=self.nodes, relationships=self.relationships,
            paths=self.paths, expected_database=self.expected_database,
        )


def fake_driver_factory(*, nodes: int, relationships: int, paths: int, expected_database: str = "neo4j"):
    return lambda *_args, **_kwargs: FakeDriver(
        nodes=nodes, relationships=relationships, paths=paths, expected_database=expected_database,
    )


def write_run(
    outputs: Path, run_id: str, *, completed_at: str | None,
    validation_passed: bool = True, complete: bool = True,
) -> Path:
    run_dir = outputs / run_id
    run_dir.mkdir(parents=True)
    graph_id = f"fact_extraction_{run_id}"
    graph = {
        "run_id": run_id,
        "graph_id": graph_id,
        "nodes": [{"entity_id": "E001"}],
        "edges": [],
    }
    (run_dir / "fact_graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (run_dir / "run_report.json").write_text(json.dumps({
        "run_id": run_id, "graph_id": graph_id,
        "neo4j": {"graph_id": graph_id, "nodes": 1, "edges": 0},
    }), encoding="utf-8")
    (run_dir / "validation_report.json").write_text(json.dumps({
        "passed": validation_passed, "json_neo4j_parity": validation_passed,
    }), encoding="utf-8")
    (run_dir / "projection_manifest.json").write_text(json.dumps({
        "manifest_version": "1",
        "graph_id": graph_id,
        "projection": {
            "entity_label": "FACT_ENTITY",
            "relation_type": "FACT_RELATION",
        },
    }), encoding="utf-8")
    if complete:
        (run_dir / "json_neo4j_edge_diff.json").write_text(json.dumps({
            "json_only_nodes": [], "neo4j_only_nodes": [], "json_only_edges": [],
            "neo4j_only_edges": [], "source_target_mismatches": [],
            "edges_without_graph_id": [], "duplicate_edges": [], "import_errors": [],
        }), encoding="utf-8")
    if completed_at:
        (run_dir / "graph_view.json").write_text(json.dumps({
            "completed_at_utc": completed_at,
        }), encoding="utf-8")
    return run_dir


def test_latest_uses_completion_metadata_and_ignores_failed_or_incomplete_runs(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    write_run(outputs, "older", completed_at="2026-01-01T00:00:00+00:00")
    write_run(outputs, "newer", completed_at="2026-02-01T00:00:00+00:00")
    write_run(outputs, "failed", completed_at="2026-03-01T00:00:00+00:00", validation_passed=False)
    write_run(outputs, "incomplete", completed_at="2026-04-01T00:00:00+00:00", complete=False)
    assert select_run(outputs).run_id == "newer"


def test_explicit_run_and_graph_id_selection(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    selected = write_run(outputs, "chosen", completed_at=None)
    assert select_run(outputs, run_id="chosen").run_dir == selected
    direct = select_run(outputs, graph_id="fact_extraction_direct")
    assert direct.graph_id == "fact_extraction_direct" and direct.run_dir is None
    with pytest.raises(ValueError, match="safe identifier"):
        select_run(outputs, graph_id='fact_extraction_x" MATCH (n)')
    with pytest.raises(ValueError, match="run_id"):
        select_run(outputs, run_id="../escape")


def test_query_filters_every_endpoint_and_relationship_and_validates_limit() -> None:
    query = graph_query("fact_extraction_safe", 321)
    assert "(source:FACT_ENTITY)-[relation:FACT_RELATION]->(target:FACT_ENTITY)" in query
    assert query.count('= "fact_extraction_safe"') == 3
    assert query.endswith("LIMIT 321")
    assert validate_limit(1) == 1 and validate_limit(5000) == 5000
    for invalid in (0, 5001):
        with pytest.raises(ValueError, match="between 1 and 5000"):
            validate_limit(invalid)


def test_browser_url_encodes_query_database_and_removes_uri_credentials() -> None:
    url = browser_url(
        base_url="http://localhost:7474/browser",
        connection_uri="bolt://user:password@localhost:7687",
        database="neo4j",
        query=graph_query("fact_extraction_safe"),
    )
    parsed = parse_qs(urlsplit(url).query)
    assert parsed["dbms"] == ["bolt://localhost:7687"]
    assert parsed["db"] == ["neo4j"]
    assert parsed["cmd"] == ["edit"]
    assert "MATCH p=" in parsed["arg"][0]
    assert "password" not in url and "user" not in url


def test_graph_view_metadata_is_secret_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://neo4j:super-secret@localhost:7687")
    monkeypatch.setenv("NEO4J_DATABASE", "neo4j")
    payload = graph_view_payload(
        GraphRun("safe", "fact_extraction_safe", tmp_path, 75, 94),
        load_project_settings(ROOT / "config"),
    )
    serialized = json.dumps(payload)
    assert payload["node_count"] == 75 and payload["relationship_count"] == 94
    assert "super-secret" not in serialized
    assert "OPENAI_API_KEY" not in serialized
    assert set(payload) == {
        "run_id", "graph_id", "browser_url", "query", "database",
        "node_count", "relationship_count", "completed_at_utc",
        "projection",
    }


def test_preflight_succeeds_for_existing_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    result = preflight_graph(
        GraphRun("safe", "fact_extraction_safe", tmp_path),
        load_project_settings(ROOT / "config"),
        driver_factory=fake_driver_factory(nodes=75, relationships=94, paths=94),
    )
    assert (result.database, result.nodes, result.relationships, result.paths) == ("neo4j", 75, 94, 94)
    assert result.filter_paths == {
        "unfiltered": 94, "source": 94, "relationship": 94, "target": 94, "all": 94,
    }


@pytest.mark.parametrize(
    ("nodes", "relationships", "paths", "message"),
    [(0, 0, 0, "graph_id does not exist"), (1, 1, 0, "zero paths")],
)
def test_preflight_rejects_unknown_graph_or_zero_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    nodes: int, relationships: int, paths: int, message: str,
) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    with pytest.raises(ViewerPreflightError, match=message):
        preflight_graph(
            GraphRun("safe", "fact_extraction_safe", tmp_path),
            load_project_settings(ROOT / "config"),
            driver_factory=fake_driver_factory(nodes=nodes, relationships=relationships, paths=paths),
        )


def test_preflight_reports_wrong_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("NEO4J_DATABASE", "wrong")
    with pytest.raises(ViewerPreflightError, match="database wrong"):
        preflight_graph(
            GraphRun("safe", "fact_extraction_safe", tmp_path),
            load_project_settings(ROOT / "config"),
            driver_factory=fake_driver_factory(nodes=75, relationships=94, paths=94),
        )


def test_browser_opener_is_called_only_when_requested_and_failure_is_nonfatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import scripts.run_production as cli

    output = tmp_path / "run_safe"
    output.mkdir()
    (output / "graph_view.json").write_text(json.dumps({"browser_url": "http://safe"}), encoding="utf-8")
    (output / "fact_graph.json").write_text(
        json.dumps({"source": {"units": [{"unit_id": "U001"}]}}),
        encoding="utf-8",
    )
    order: list[str] = []
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: order.append("run") or 0)
    monkeypatch.setattr(cli, "load_successful_run", lambda path: GraphRun(
        "run_safe", "fact_extraction_run_safe", path, 1, 2,
    ))
    monkeypatch.setattr(cli, "preflight_graph", lambda *_args, **_kwargs: (
        order.append("preflight") or GraphPreflight("neo4j", 1, 2, 2, {"all": 2})
    ))
    monkeypatch.setattr(cli, "open_browser", lambda url: (order.append("open") or False, "blocked"))
    monkeypatch.setattr(sys, "argv", [
        "run_production", "--input", str(tmp_path / "source.txt"),
        "--output", str(output), "--open-graph",
    ])
    assert cli.main() == 0
    assert order == ["run", "preflight", "open"]
    output_text = capsys.readouterr()
    assert "Graph ID: fact_extraction_run_safe" in output_text.out
    assert "python -m scripts.open_graph --run-id run_safe --open" in output_text.out
    assert "Warning: blocked" in output_text.err

    order.clear()
    monkeypatch.setattr(sys, "argv", [
        "run_production", "--input", str(tmp_path / "source.txt"), "--output", str(output),
    ])
    assert cli.main() == 0
    assert order == ["run"]


def test_production_cli_never_opens_when_projection_or_parity_is_not_successful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.run_production as cli

    output = tmp_path / "failed_run"
    order: list[str] = []
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: order.append("run") or 0)
    monkeypatch.setattr(
        cli, "load_successful_run",
        lambda _path: (_ for _ in ()).throw(ValueError("run parity failed")),
    )
    monkeypatch.setattr(cli, "open_browser", lambda _url: order.append("open"))
    monkeypatch.setattr(sys, "argv", [
        "run_production", "--input", str(tmp_path / "source.txt"),
        "--output", str(output), "--open-graph",
    ])
    assert cli.main() == 0
    assert order == ["run"]


def test_open_browser_catches_system_failure() -> None:
    opened, warning = open_browser(
        "http://safe",
        opener=lambda _url: (_ for _ in ()).throw(OSError("no browser")),
    )
    assert opened is False and warning == "OSError: no browser"


def test_open_graph_cli_opens_only_with_open_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import scripts.open_graph as cli

    outputs = tmp_path / "outputs"
    write_run(outputs, "chosen", completed_at="2026-01-01T00:00:00+00:00")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    opened: list[str] = []
    monkeypatch.setattr(cli, "open_browser", lambda url: (opened.append(url) or True, None))
    monkeypatch.setattr(cli, "preflight_graph", lambda *_args, **_kwargs: GraphPreflight(
        "neo4j", 1, 1, 1, {"all": 1},
    ))
    monkeypatch.setattr(sys, "argv", [
        "open_graph", "--run-id", "chosen", "--print-only", "--show-query",
        "--outputs-dir", str(outputs), "--config-dir", str(ROOT / "config"),
    ])
    assert cli.main() == 0 and opened == []
    output = capsys.readouterr().out
    assert "Preflight nodes: 1" in output
    assert "Preflight relationships: 1" in output
    assert "Preflight paths: 1" in output
    assert "Relationship caption: raw_relation" in output
    assert "Browser stylesheet: config/neo4j_browser.grass" in output
    assert "Browser style is not installed in this Browser session." in output
    assert 'MATCH p=(source:FACT_ENTITY)-[relation:FACT_RELATION]->(target:FACT_ENTITY)' in output
    assert "secret" not in output
    monkeypatch.setattr(sys, "argv", [
        "open_graph", "--run-id", "chosen", "--open",
        "--outputs-dir", str(outputs), "--config-dir", str(ROOT / "config"),
    ])
    assert cli.main() == 0 and len(opened) == 1


def test_open_graph_cli_does_not_open_browser_when_preflight_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import scripts.open_graph as cli

    outputs = tmp_path / "outputs"
    write_run(outputs, "chosen", completed_at="2026-01-01T00:00:00+00:00")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    opened: list[str] = []
    monkeypatch.setattr(cli, "open_browser", lambda url: (opened.append(url) or True, None))
    monkeypatch.setattr(
        cli,
        "preflight_graph",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ViewerPreflightError(
                "viewer preflight returned zero paths",
                diagnostics={"nodes": 1, "relationships": 1, "paths": 0},
            )
        ),
    )
    monkeypatch.setattr(sys, "argv", [
        "open_graph", "--run-id", "chosen", "--open",
        "--outputs-dir", str(outputs), "--config-dir", str(ROOT / "config"),
    ])
    assert cli.main() == 2
    assert opened == []
    error = capsys.readouterr().err
    assert "viewer preflight returned zero paths" in error
    assert "nodes: 1" in error and "paths: 0" in error
