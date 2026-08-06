from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_neo4j_example_uses_real_projection_and_read_only_namespace_query() -> None:
    page = (ROOT / "docs" / "example-graph.md").read_text(encoding="utf-8")
    assert "actual Neo4j projection" in page
    assert "not separately reconstructed" in page
    assert "MATCH (source:FACT_ENTITY)-[relation:FACT_RELATION]->(target:FACT_ENTITY)" in page
    for owner in ("source", "relation", "target"):
        assert f"{owner}.graph_id = $graph_id" in page
    assert "RETURN source, relation, target" in page
    assert "canonical_name" in page
    assert "raw_relation" in page
    assert "projection_manifest.json" in page


def test_readme_links_to_neo4j_example_without_placeholder_screenshot() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "[the Neo4j graph example](docs/example-graph.md)" in readme
    assert "not a separate visualization implementation" in readme
    screenshot = ROOT / "docs" / "images" / "example-neo4j-graph.png"
    if "docs/images/example-neo4j-graph.png" in readme:
        assert screenshot.is_file()
    else:
        assert not screenshot.exists()


def test_neo4j_example_relative_links_exist() -> None:
    page_path = ROOT / "docs" / "example-graph.md"
    page = page_path.read_text(encoding="utf-8")
    targets = re.findall(r"\[[^\]]+\]\((?!https?://|#)([^)]+)\)", page)
    assert targets
    assert all(
        (page_path.parent / target.split("#", 1)[0]).resolve().exists()
        for target in targets
    )


def test_interrupted_task_added_no_custom_renderer() -> None:
    assert not (ROOT / "scripts" / "render_example_graph.py").exists()
