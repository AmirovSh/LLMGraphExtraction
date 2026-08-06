from __future__ import annotations

import json
import hashlib
import re
import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from config.settings import ProjectSettings, load_project_settings
from prompts.contracts import SpanSemanticTemporalExtraction
from prompts.registry import resolve_prompt
from runtime.production_inputs import request_payload
from runtime.json_neo4j_parity import (
    import_neo4j,
    neo4j_relationship_properties,
    validate_relationship_raw_relation,
)
from runtime.transport_recovery import parse_tool_output


ROOT = Path(__file__).resolve().parents[1]


def test_prompt_registry_resolves_public_identity_contract_and_hash() -> None:
    bundle = resolve_prompt("prompt_kimi_default")
    assert bundle.prompt_id == "prompt_kimi_default"
    assert bundle.contract is SpanSemanticTemporalExtraction
    assert bundle.content_hash == "fb95eee07c769423a9ef64796c0aa302a96c0161929b28d6389a822fbd1d9d43"
    assert hashlib.sha256(
        json.dumps(bundle.schema, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest() == "eb4ce676614fd8e107c44563a487e0c16ba9d95ee047109bfc0d05f8c00ed3ae"
    assert bundle.schema["type"] == "object"
    assert len(bundle.content_hash) == 64
    assert bundle.content_hash == resolve_prompt("prompt_kimi_default").content_hash
    assert bundle.schema["additionalProperties"] is False
    assert set(bundle.schema["required"]) == {
        "status", "entities", "relations", "temporal_bindings",
        "no_graph_fact_reason",
    }
    assert "entities" in bundle.schema_builder(
        max_entities=60,
        max_relations=40,
        max_evidence_spans=10,
        max_relation_description_characters=300,
    )["properties"]
    with pytest.raises(ValueError, match="unknown prompt_id"):
        resolve_prompt("unknown")


@pytest.mark.parametrize(
    ("contract_id", "prompt_id"),
    [
        ("unknown", "prompt_kimi_default"),
        ("evidence_span_fact_extraction", "unknown_prompt"),
        ("unsupported_contract", "prompt_kimi_default"),
    ],
)
def test_unknown_prompt_config_fails_during_loading(
    tmp_path: Path, contract_id: str, prompt_id: str,
) -> None:
    config_dir = tmp_path / "config"
    shutil.copytree(ROOT / "config", config_dir)
    path = config_dir / "extraction.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["extraction"]["contract_id"] = contract_id
    payload["extraction"]["prompt_id"] = prompt_id
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="extraction_contracts|unknown prompt_id"):
        load_project_settings(config_dir)


def test_contract_selection_is_configured_and_model_independent() -> None:
    settings = load_project_settings(ROOT / "config")
    selected = resolve_prompt(settings.extraction.prompt_id)
    assert selected.contract is SpanSemanticTemporalExtraction
    values = settings.model_dump()
    values["runtime"]["remote"]["provider_profiles"][
        "kimi_k2_6_vllm_structured"
    ]["model_id"] = "provider-model-renamed"
    renamed = ProjectSettings.model_validate(values)
    assert resolve_prompt(
        renamed.extraction.prompt_id
    ).contract is SpanSemanticTemporalExtraction


def test_generic_provider_profiles_and_request_override_forwarding() -> None:
    kimi = load_project_settings(ROOT / "config")
    kimi_payload = request_payload(kimi, "extract", {"type": "object"})
    assert kimi_payload["model"] == "kimi-2.6"
    assert kimi_payload["chat_template_kwargs"] == {
        "thinking": False, "preserve_thinking": False,
    }
    assert "enable_thinking" not in kimi_payload["chat_template_kwargs"]
    assert kimi_payload["tool_choice"] == "auto"
    assert kimi_payload["parallel_tool_calls"] is False
    assert kimi_payload["temperature"] == 0 and "top_p" not in kimi_payload
    assert "seed" not in kimi_payload

    values = kimi.model_dump()
    values["runtime"]["remote"]["llm_provider_profile"] = (
        "kimi_k2_6_vllm_instant"
    )
    general_payload = request_payload(
        ProjectSettings.model_validate(values), "extract", {"type": "object"},
    )
    assert (
        general_payload["temperature"] == .6
        and general_payload["top_p"] == .95
    )

    values = kimi.model_dump()
    values["runtime"]["remote"]["llm_provider_profile"] = (
        "kimi_k2_6_vllm_structured"
    )
    structured_payload = request_payload(
        ProjectSettings.model_validate(values), "extract", {"type": "object"},
    )
    assert structured_payload["temperature"] == 0
    assert "top_p" not in structured_payload
    assert "seed" not in structured_payload
    assert structured_payload["tool_choice"] == "auto"

    values = kimi.model_dump()
    values["runtime"]["remote"]["llm_provider_profile"] = (
        "kimi_k2_6_vllm_thinking_structured"
    )
    thinking = ProjectSettings.model_validate(values)
    thinking_profile = thinking.runtime.remote.selected_provider_profile
    thinking_payload = request_payload(
        thinking, "extract", {"type": "object"},
    )
    assert thinking_payload["chat_template_kwargs"] == {
        "thinking": True, "preserve_thinking": False,
    }
    assert thinking_payload["temperature"] == 1.0
    assert thinking_payload["top_p"] == 0.95
    assert thinking_payload["max_tokens"] == 32768
    assert thinking_profile.timeout_seconds == 900
    assert thinking_profile.reasoning_policy == "allow_nonempty"
    assert thinking_profile.reasoning_persistence == "discard"

    values = kimi.model_dump()
    values["runtime"]["remote"]["provider_profiles"][
        "kimi_k2_6_vllm_structured"
    ]["request_extra_body"] = {"vendor": {"nested": {"enabled": True}}}
    with pytest.raises(ValidationError, match="unsupported extension key"):
        ProjectSettings.model_validate(values)

    values = kimi.model_dump()
    values["runtime"]["remote"]["llm_request_overrides"] = {"top_p": 0.75}
    with pytest.raises(ValidationError, match="unsupported extension key"):
        ProjectSettings.model_validate(values)

    values = kimi.model_dump()
    values["runtime"]["remote"]["llm_provider_profile"] = "unsupported"
    with pytest.raises(ValidationError, match="llm_provider_profile"):
        ProjectSettings.model_validate(values)


def test_generic_tool_call_parsing_has_no_provider_assumption() -> None:
    arguments = {"entities": [], "relations": []}
    parsed, transport = parse_tool_output(
        {"tool_calls": [{"function": {
            "name": "submit_dynamic_inventory",
            "arguments": json.dumps(arguments),
        }}]}
    )
    assert parsed == arguments
    assert transport == "native_tool_call"
    with pytest.raises(RuntimeError, match="wrong function"):
        parse_tool_output(
            {"tool_calls": [{"function": {
                "name": "other",
                "arguments": json.dumps(arguments),
            }}]}
        )


def test_required_agent_and_contributor_documents_exist() -> None:
    required = [
        "AGENTS.md",
        ".agents/README.md",
        ".agents/policies/architecture.md",
        ".agents/policies/security-and-artifacts.md",
        ".agents/policies/text-processing.md",
        ".agents/policies/model-connections.md",
        ".agents/policies/long-running-work.md",
        ".agents/templates/task.md",
        ".agents/templates/review-report.md",
        "CONTRIBUTING.md",
    ]
    assert all((ROOT / path).is_file() for path in required)


def test_public_neo4j_schema_uses_only_neutral_names() -> None:
    from config.projection_identifiers import DEFAULT_PROJECTION_TYPES

    source = (
        ROOT / "config" / "projection_identifiers.py"
    ).read_text(encoding="utf-8")
    for expected in ("FACT_ENTITY", "FACT_RELATION"):
        assert expected in source
    assert DEFAULT_PROJECTION_TYPES.expected_indexes == {
        "fact_entity_graph_id", "fact_relation_graph_id",
    }


def test_neo4j_relationship_requires_and_preserves_raw_relation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edge = {
        "edge_id": "R_test",
        "source_entity_id": "E001",
        "target_entity_id": "E002",
        "raw_relation": "  preserves exact wording  ",
        "qualifiers": {},
        "provenance": [],
    }
    assert neo4j_relationship_properties(edge, "fact_extraction_test")["raw_relation"] == (
        "  preserves exact wording  "
    )
    for invalid in (None, "", "   ", 17):
        invalid_edge = {**edge, "raw_relation": invalid}
        with pytest.raises(ValueError, match="raw_relation must be a non-empty string"):
            validate_relationship_raw_relation(invalid_edge)

    monkeypatch.setattr(
        "runtime.json_neo4j_parity.GraphDatabase.driver",
        lambda *_args, **_kwargs: pytest.fail("Neo4j write attempted"),
    )
    with pytest.raises(ValueError, match="R_test: raw_relation must be a non-empty string"):
        import_neo4j(
            "fact_extraction_test",
            [{"entity_id": "E001"}, {"entity_id": "E002"}],
            [{**edge, "raw_relation": " "}],
            load_project_settings(ROOT / "config"),
            {"run_id": "test", "source_sha256": "a" * 64, "manifest_identity_hash": "b" * 64},
        )


def test_neo4j_browser_stylesheet_uses_raw_relation_caption() -> None:
    stylesheet = (ROOT / "config" / "neo4j_browser.grass").read_text(encoding="utf-8")
    assert "relationship.FACT_RELATION" in stylesheet
    assert 'caption: "{raw_relation}";' in stylesheet
    assert 'node.FACT_ENTITY' in stylesheet
    assert 'caption: "{canonical_name}";' in stylesheet
    assert "PromptV2" not in stylesheet
    assert "password" not in stylesheet.casefold()


def test_readme_local_links_resolve() -> None:
    content = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\[[^\]]+\]\((?!https?://|#)([^)]+)\)", content)
    assert all((ROOT / target.split("#", 1)[0]).exists() for target in targets)
