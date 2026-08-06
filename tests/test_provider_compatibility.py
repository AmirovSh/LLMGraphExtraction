from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from config.settings import ProjectSettings, load_project_settings
from prompts.contracts import FactExtraction
from pydantic import BaseModel

from runtime.provider_compatibility import (
    apply_thinking_override,
    discard_reasoning_text,
    evaluate_structured_response,
)
from devtools.probe_model_compatibility import _semantic_assertions
from devtools.probe_model_compatibility import main as probe_main

ROOT = Path(__file__).resolve().parents[1]


class _Item(BaseModel):
    evidence_span_ids: list[str]


class _Extraction(BaseModel):
    entities: list[_Item]
    relations: list[_Item]


def test_thinking_override_is_explicit_and_does_not_mutate_payload() -> None:
    source = {"messages": []}
    result = apply_thinking_override(source, "chat_template_kwargs")
    assert result["chat_template_kwargs"] == {"enable_thinking": False}
    assert source == {"messages": []}


def test_structured_response_reports_schema_and_evidence_without_content() -> None:
    arguments = {
        "entities": [{"evidence_span_ids": ["S001"]}],
        "relations": [],
    }
    body = {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "tool_calls": [{
                    "function": {
                        "name": "submit_dynamic_inventory",
                        "arguments": json.dumps(arguments),
                    }
                }]
            },
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    }
    result = evaluate_structured_response(
        transport={
            "http_status": 200,
            "duration_seconds": 0.1,
            "response_json": True,
            "exception_type": None,
        },
        body=body,
        expected_function="submit_dynamic_inventory",
        contract=_Extraction,
        valid_span_ids={"S001"},
    )
    assert result["result"] == "PASS"
    assert result["tool_calls"] == 1
    assert result["evidence_valid"] is True
    assert "arguments" not in result
    assert result["entity_count"] == 1
    assert result["relation_count"] == 0
    assert len(result["response_argument_hash"]) == 64


def test_safe_diagnostics_hash_predicates_and_count_families() -> None:
    arguments = {
        "entities": [{"evidence_span_ids": ["S001"]}],
        "relations": [{
            "evidence_span_ids": ["S001"],
            "relation_family": "DATA_FLOW",
            "raw_relation": "Reads  Records From",
        }],
    }
    body = {"choices": [{"message": {"tool_calls": [{"function": {
        "name": "submit_dynamic_inventory",
        "arguments": json.dumps(arguments),
    }}]}}]}
    result = evaluate_structured_response(
        transport={"http_status": 200}, body=body,
        expected_function="submit_dynamic_inventory", contract=_Extraction,
        valid_span_ids={"S001"}, reasoning_policy="ignore",
    )
    assert result["relation_family_multiset"] == ["DATA_FLOW"]
    assert result["raw_relation_normalized_hashes"] == [
        __import__("hashlib").sha256(b"reads records from").hexdigest()
    ]
    assert "Reads  Records From" not in json.dumps(result)


def test_structured_response_rejects_unknown_evidence() -> None:
    body = {
        "choices": [{
            "message": {
                "tool_calls": [{
                    "function": {
                        "name": "submit_dynamic_inventory",
                        "arguments": json.dumps({
                            "entities": [{"evidence_span_ids": ["S999"]}],
                            "relations": [],
                        }),
                    }
                }]
            }
        }]
    }
    result = evaluate_structured_response(
        transport={
            "http_status": 200,
            "duration_seconds": 0.1,
            "response_json": True,
            "exception_type": None,
        },
        body=body,
        expected_function="submit_dynamic_inventory",
        contract=_Extraction,
        valid_span_ids={"S001"},
    )
    assert result["result"] == "FAIL"
    assert result["pydantic_valid"] is True
    assert result["evidence_valid"] is False


def test_thinking_response_is_accepted_and_reasoning_is_discarded() -> None:
    body = {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "reasoning_content": "private chain of thought",
                "content": None,
                "tool_calls": [{"function": {
                    "name": "submit_dynamic_inventory",
                    "arguments": json.dumps({
                        "entities": [{"evidence_span_ids": ["S001"]}],
                        "relations": [],
                    }),
                }}],
            },
        }],
        "usage": {
            "completion_tokens_details": {"reasoning_tokens": 17},
        },
    }
    result = evaluate_structured_response(
        transport={"http_status": 200},
        body=body,
        expected_function="submit_dynamic_inventory",
        contract=_Extraction,
        valid_span_ids={"S001"},
        reasoning_policy="allow_nonempty",
        accepted_finish_reasons=["tool_calls", "stop"],
    )
    sanitized, metadata = discard_reasoning_text(
        body, ["reasoning", "reasoning_content"],
    )
    assert result["result"] == "PASS"
    assert result["reasoning_present"] is True
    assert result["reasoning_separated"] is True
    assert metadata == {"reasoning_present": True, "reasoning_tokens": 17}
    assert "reasoning_content" not in sanitized["choices"][0]["message"]
    assert "private chain of thought" not in json.dumps(sanitized)


def test_thinking_response_rejects_reasoning_leaked_into_content() -> None:
    result = evaluate_structured_response(
        transport={"http_status": 200},
        body={"choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "reasoning_content": "separate",
                "content": "<think>leaked</think>",
                "tool_calls": [{"function": {
                    "name": "submit_dynamic_inventory",
                    "arguments": json.dumps({"entities": [], "relations": []}),
                }}],
            },
        }]},
        expected_function="submit_dynamic_inventory",
        contract=_Extraction,
        valid_span_ids={"S001"},
        reasoning_policy="allow_nonempty",
    )
    assert result["result"] == "FAIL"
    assert result["raw_special_tokens_present"] is True


def test_structured_response_classifies_missing_fields_and_invalid_ids() -> None:
    body = {
        "choices": [{
            "message": {
                "tool_calls": [{
                    "function": {
                        "name": "submit_dynamic_inventory",
                        "arguments": json.dumps({
                            "entities": [{
                                "local_id": "M001,",
                                "name": "Entity",
                                "type": "OBJECT",
                                "evidence_span_ids": ["S001"],
                            }],
                            "relations": [{
                                "source_local_id": "M001,",
                                "target_local_id": "M002,",
                            }],
                        }),
                    }
                }]
            }
        }]
    }
    result = evaluate_structured_response(
        transport={
            "http_status": 200,
            "duration_seconds": 0.1,
            "response_json": True,
            "exception_type": None,
        },
        body=body,
        expected_function="submit_dynamic_inventory",
        contract=FactExtraction,
        valid_span_ids={"S001"},
    )
    assert result["result"] == "FAIL"
    assert result["schema_failure_categories"] == [
        "INVALID_LOCAL_ID", "MISSING_REQUIRED_FIELD",
    ]
    assert any(
        issue["type"] == "missing" and issue["path"].endswith("raw_relation")
        for issue in result["schema_issues"]
    )
    assert result["structure"]["entities_is_array"] is True
    assert result["structure"]["relations_is_array"] is True


def test_runtime_request_builder_has_no_model_name_routing() -> None:
    source = (ROOT / "runtime" / "production_inputs.py").read_text(
        encoding="utf-8"
    ).casefold()
    assert "kimi" not in source
    assert "deepseek" not in source
    assert "model_name.lower" not in source


def test_profile_environment_json_overlay_uses_typed_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_project_settings(ROOT / "config")
    profile = settings.runtime.remote.selected_provider_profile.model_dump()
    profile["profile_id"] = "environment_profile"
    profile["model_id"] = "environment-model"
    monkeypatch.setenv(
        "SEMANTIC_GRAPH_PROVIDER_PROFILES_JSON",
        json.dumps({"environment_profile": profile}),
    )
    monkeypatch.setenv(
        "SEMANTIC_GRAPH_LLM_PROVIDER_PROFILE", "environment_profile",
    )
    loaded = load_project_settings(ROOT / "config")
    assert loaded.runtime.remote.selected_provider_profile.model_id == (
        "environment-model"
    )


def test_profile_rejects_credential_fields() -> None:
    values = load_project_settings(ROOT / "config").model_dump()
    values["runtime"]["remote"]["provider_profiles"][
        "kimi_k2_6_vllm_instant"
    ]["request_extra_body"] = {"Authorization": "must-not-be-persisted"}
    with pytest.raises(ValidationError, match="credential-like key"):
        ProjectSettings.model_validate(values)


@pytest.mark.parametrize("reserved", ["model", "messages", "tools"])
def test_run_request_overrides_reject_contract_keys(reserved: str) -> None:
    values = load_project_settings(ROOT / "config").model_dump()
    values["runtime"]["remote"]["llm_request_overrides"] = {reserved: "changed"}
    with pytest.raises(ValidationError, match="reserved or credential-like key"):
        ProjectSettings.model_validate(values)


def test_run_request_overrides_reject_nested_credential_variants() -> None:
    values = load_project_settings(ROOT / "config").model_dump()
    values["runtime"]["remote"]["llm_request_overrides"] = {
        "chat_template_kwargs": {"Api-Key": "secret"},
    }
    with pytest.raises(ValidationError, match="reserved or credential-like key"):
        ProjectSettings.model_validate(values)


def test_run_request_overrides_allow_only_supported_chat_template_options() -> None:
    values = load_project_settings(ROOT / "config").model_dump()
    values["runtime"]["remote"]["llm_request_overrides"] = {
        "chat_template_kwargs": {"thinking": False, "preserve_thinking": False},
    }
    settings = ProjectSettings.model_validate(values)
    assert settings.runtime.remote.llm_request_overrides == {
        "chat_template_kwargs": {"thinking": False, "preserve_thinking": False},
    }


def test_retired_qwen_benchmark_profile_is_not_selectable() -> None:
    with pytest.raises(ValueError, match="unknown llm_provider_profile"):
        load_project_settings(
            ROOT / "config",
            overrides={
                "runtime.remote.llm_provider_profile": (
                    "qwen3_coder_next_vllm_structured"
                ),
            },
        )


def test_semantic_probe_ignores_relations_from_other_units() -> None:
    body = {
        "choices": [{
            "message": {
                "tool_calls": [{
                    "function": {
                        "arguments": json.dumps({
                            "entities": [],
                            "relations": [],
                        }),
                    }
                }]
            }
        }]
    }
    manifest = {
        "required_relations": [{
            "source": "Scheduling Service",
            "raw_relation": "creates",
            "target": "Maintenance Work Order",
            "negated": False,
        }],
        "forbidden_relations": [],
        "required_qualifier_cases": [],
    }
    result = _semantic_assertions(
        body,
        manifest=manifest,
        unit_text="The hub sends an order to the Scheduling Service.",
    )
    assert result == {"passed": True, "failures": []}


def test_semantic_probe_supports_compact_index_relations_and_bindings() -> None:
    arguments = {
        "entities": [
            {"canonical_name": "Policy", "entity_type": "CONTROL"},
            {"canonical_name": "applicability", "entity_type": "CONCEPT"},
        ],
        "relations": [{
            "source_entity_index": 0, "target_entity_index": 1,
            "raw_relation": "applies", "negated": False,
        }],
        "temporal_bindings": [{"relation_index": 0, "surface": "during 2027"}],
    }
    body = {"choices": [{"message": {"tool_calls": [{"function": {
        "arguments": json.dumps(arguments),
    }}]}}]}
    manifest = {
        "required_relations": [{
            "source": "Policy", "raw_relation": "applies",
            "target": "applicability", "negated": False,
        }],
        "required_qualifier_cases": [{
            "source": "Policy", "raw_relation": "applies",
            "field": "temporality", "value": "during 2027",
        }],
    }
    assert _semantic_assertions(
        body, manifest=manifest,
        unit_text="Policy applies to applicability during 2027.",
    ) == {"passed": True, "failures": []}


def test_production_only_probe_requires_production_repetitions() -> None:
    with pytest.raises(SystemExit) as captured:
        probe_main([
            "--profile", "kimi_k2_6_vllm_structured",
            "--production-only",
        ])
    assert captured.value.code == 2


def test_probe_prompt_override_requires_explicit_diagnostic_identity() -> None:
    with pytest.raises(SystemExit) as captured:
        probe_main([
            "--profile", "kimi_k2_6_vllm_structured",
            "--prompt-template", "probe.md",
        ])
    assert captured.value.code == 2
