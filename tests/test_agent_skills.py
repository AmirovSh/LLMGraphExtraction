from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / ".agents"
SKILLS = AGENTS / "skills"
NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
ABSOLUTE_PATH_PATTERN = re.compile(r"(?:[A-Za-z]:\\Users\\|/Users/|/home/)")
SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(?:api[_-]?key|password|secret|authorization)\s*[:=]\s*[\"']?(?!<|\\{|\\$)[^\s\"']+"
)
OLD_SKILL_PATHS = tuple(
    name.replace("-", "_")
    for name in (
        "architecture-review", "fact-graph-model-review", "llm-pipeline-review",
        "local-model-connectivity", "regression-golden-review",
        "schema-contract-review", "text-processing-audit",
    )
)
ALLOWED_FRONTMATTER = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}


def skill_directories() -> list[Path]:
    return sorted(path for path in SKILLS.iterdir() if path.is_dir())


def parse_skill(path: Path) -> tuple[dict, str]:
    content = path.read_text(encoding="utf-8")
    assert content.startswith("---\n"), f"{path} must start with YAML frontmatter"
    closing = content.find("\n---\n", 4)
    assert closing >= 0, f"{path} has unterminated YAML frontmatter"
    metadata = yaml.safe_load(content[4:closing])
    assert isinstance(metadata, dict)
    return metadata, content[closing + 5:]


def test_agent_skill_structure_and_frontmatter() -> None:
    directories = skill_directories()
    assert {path.name for path in directories} == {
        "architecture-review", "fact-graph-model-review", "llm-pipeline-review",
        "local-model-connectivity", "regression-golden-review",
        "schema-contract-review", "text-processing-audit",
    }
    for directory in directories:
        skill_file = directory / "SKILL.md"
        assert skill_file.is_file()
        metadata, body = parse_skill(skill_file)
        assert set(metadata) <= ALLOWED_FRONTMATTER
        assert metadata["name"] == directory.name
        assert NAME_PATTERN.fullmatch(metadata["name"])
        assert len(metadata["name"]) <= 64
        description = metadata.get("description")
        assert isinstance(description, str) and description.strip()
        assert len(description) <= 1024
        assert "use when" in description.casefold()
        assert len(skill_file.read_text(encoding="utf-8").splitlines()) <= 500
        assert body.strip()


def test_agent_skill_links_resources_and_scripts_are_valid() -> None:
    for directory in skill_directories():
        skill_file = directory / "SKILL.md"
        content = skill_file.read_text(encoding="utf-8")
        for target in LINK_PATTERN.findall(content):
            if "://" in target or target.startswith("#"):
                continue
            assert (directory / target.split("#", 1)[0]).resolve().exists(), (
                f"broken link in {skill_file}: {target}"
            )
        for reference in (directory / "references").glob("*"):
            assert reference.is_file() and reference.stat().st_size > 0
        scripts = list((directory / "scripts").glob("*")) if (directory / "scripts").exists() else []
        for script in scripts:
            assert script.name in content or any(
                script.name in reference.read_text(encoding="utf-8")
                for reference in (directory / "references").glob("*")
            ), f"{script} is not referenced"


def test_public_agent_text_has_no_private_or_obsolete_content() -> None:
    files = [ROOT / "AGENTS.md", *AGENTS.rglob("*")]
    text_files = [path for path in files if path.is_file()]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in text_files)
    assert not ABSOLUTE_PATH_PATTERN.search(combined)
    assert not SECRET_ASSIGNMENT_PATTERN.search(combined)
    assert "PromptV2" not in combined
    assert "PROMPT_V2" not in combined
    assert "prompt_v2" not in combined
    assert not any(value in combined for value in OLD_SKILL_PATHS)
    assert "release_audit/" not in combined


def test_agent_supporting_files_are_nonempty_and_tracked_shape_is_public() -> None:
    expected = {
        AGENTS / "README.md",
        *(AGENTS / "policies").glob("*.md"),
        *(AGENTS / "templates").glob("*.md"),
    }
    assert expected
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected)
    assert not (AGENTS / ".codex").exists()
