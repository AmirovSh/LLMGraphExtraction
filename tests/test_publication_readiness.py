from __future__ import annotations

from pathlib import Path


def test_publication_metadata_and_local_compose_defaults() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    for heading in (
        "Project status",
        "What the project does",
        "Architecture",
        "Pipeline stages",
        "Requirements",
        "Installation",
        "Configuration",
        "Environment variables",
        "Running extraction",
        "Output artifacts",
        "Neo4j projection",
        "Example graph",
        "Testing",
        "Quality guarantees",
        "Known limitations",
        "Privacy and data retention",
        "Development",
        "License",
    ):
        assert f"## {heading}" in readme
    assert "Status: Experimental Release Candidate" in readme
    assert "OWNER LICENSE DECISION REQUIRED" in readme

    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.1.0rc1"' in pyproject
    assert 'dev = ["pytest==9.0.2"]' in pyproject

    compose = (root / "docker-compose.neo4j.yml").read_text(encoding="utf-8")
    assert '"127.0.0.1:7474:7474"' in compose
    assert '"127.0.0.1:7687:7687"' in compose
    assert "healthcheck:" in compose
    assert "APOC" not in compose.upper()

    assert (root / "CONTRIBUTING.md").is_file()
    assert (root / "SECURITY.md").is_file()
    assert (root / "CHANGELOG.md").is_file()
    assert not (root / "LICENSE").exists()


def test_production_and_devtool_source_boundaries() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in (
        root / "runtime" / "public_golden.py",
        root / "runtime" / "project_conformance.py",
        root / "scripts" / "check_public_golden.py",
        root / "scripts" / "check_project_conformance.py",
        root / "scripts" / "probe_model_compatibility.py",
        root / "scripts" / "probe_kimi26_compatibility.py",
        root / "scripts" / "render_example_graph.py",
    ):
        assert not path.exists()
    for path in (
        root / "devtools" / "public_golden.py",
        root / "devtools" / "project_conformance.py",
        root / "devtools" / "check_public_golden.py",
        root / "devtools" / "check_project_conformance.py",
        root / "devtools" / "probe_model_compatibility.py",
    ):
        assert path.is_file()
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'include = ["config*", "prompts*", "runtime*", "scripts*"]' in pyproject
    assert "devtools*" not in pyproject
    assert 'config = ["*.yaml", "*.grass"]' in pyproject
    assert '"prompts.fact_extraction" = ["*.md", "*.yaml"]' in pyproject
