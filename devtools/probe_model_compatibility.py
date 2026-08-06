"""Probe a configured provider profile without persistence, retries, or fallbacks."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import environment_value, load_project_settings
from prompts.contracts import SpanSemanticTemporalExtraction
from prompts.registry import resolve_prompt
from runtime.production_inputs import (
    render_prompt,
    request_payload,
    sentence_spans,
    split_unit_spans,
)
from runtime.provider_compatibility import (
    evaluate_structured_response,
    execute_probe,
)
from runtime.span_extraction import ExtractionContractViolation
from runtime.span_semantic_temporal_contract import (
    validate_and_enrich_semantic_temporal,
)


def _profile_payload(
    profile: Any, *, messages: list[dict[str, str]], max_tokens: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": profile.model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": profile.stream,
    }
    for key, value in (
        ("temperature", profile.temperature),
        ("top_p", profile.top_p),
        ("seed", profile.seed),
    ):
        if value is not None:
            payload[key] = value
    payload.update(json.loads(json.dumps(profile.request_extra_body)))
    return payload


def _message(body: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    choices = (body or {}).get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    return choice, message


def _plain_result(
    transport: dict[str, Any], body: dict[str, Any] | None, profile: Any,
) -> dict[str, Any]:
    choice, message = _message(body)
    usage = (body or {}).get("usage") or {}
    reasoning_absent = not any(
        message.get(field) for field in profile.reasoning_response_fields
    )
    finish = choice.get("finish_reason")
    reasoning_present = not reasoning_absent
    reasoning_valid = (
        reasoning_absent if profile.reasoning_policy == "require_empty"
        else reasoning_present if profile.reasoning_policy == "allow_nonempty"
        else True
    )
    passed = (
        transport.get("http_status") == 200
        and (bool(message.get("content")) or reasoning_present)
        and reasoning_valid
        and finish != "length"
    )
    return {
        **transport,
        "content_present": bool(message.get("content")),
        "reasoning_absent": reasoning_absent,
        "reasoning_present": reasoning_present,
        "finish_reason": finish,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "result": "PASS" if passed else "FAIL",
        "classification": (
            None if passed else
            "CHAT_TEMPLATE_KWARGS_IGNORED_OR_UNSUPPORTED"
            if not reasoning_valid else
            "OUTPUT_BUDGET_EXHAUSTED"
            if finish == "length" else
            "PLAIN_CHAT_FAILED"
        ),
    }


def _minimal_tool_result(
    transport: dict[str, Any], body: dict[str, Any] | None, profile: Any,
) -> dict[str, Any]:
    choice, message = _message(body)
    calls = message.get("tool_calls") or []
    reasoning_absent = not any(
        message.get(field) for field in profile.reasoning_response_fields
    )
    correct = (
        len(calls) == 1
        and (calls[0].get("function") or {}).get("name") == "return_value"
    )
    arguments_object = False
    if correct:
        try:
            arguments_object = isinstance(
                json.loads(calls[0]["function"]["arguments"]), dict
            )
        except (KeyError, TypeError, json.JSONDecodeError):
            pass
    finish = choice.get("finish_reason")
    reasoning_present = not reasoning_absent
    reasoning_separated = reasoning_present and not bool(message.get("content"))
    reasoning_valid = (
        reasoning_absent if profile.reasoning_policy == "require_empty"
        else reasoning_separated if profile.reasoning_policy == "allow_nonempty"
        else True
    )
    content = message.get("content")
    raw_special_tokens_present = isinstance(content, str) and any(
        token in content for token in ("<think>", "</think>", "<|tool_call_")
    )
    passed = (
        transport.get("http_status") == 200
        and correct
        and arguments_object
        and reasoning_valid
        and not raw_special_tokens_present
        and finish in profile.accepted_finish_reasons
    )
    if passed:
        classification = None
    elif transport.get("http_status") == 500:
        classification = "SERVER_TOOL_CHOICE_MODE_UNSUPPORTED"
    elif finish == "length":
        classification = "OUTPUT_BUDGET_EXHAUSTED"
    elif not reasoning_absent:
        classification = "THINKING_DISABLE_PARAMETER_IGNORED"
    elif not calls:
        classification = (
            "SERVER_KIMI_TOOL_PARSER_MISSING"
            if isinstance(message.get("content"), str)
            and "<|tool_call_" in message["content"]
            else "MODEL_DID_NOT_CALL_TOOL"
        )
    else:
        classification = "TOOL_PARSER_OR_MODEL_OUTPUT_INVALID"
    usage = (body or {}).get("usage") or {}
    return {
        **transport,
        "finish_reason": finish,
        "tool_calls": len(calls),
        "correct_function": correct,
        "arguments_json_object": arguments_object,
        "reasoning_absent": reasoning_absent,
        "reasoning_present": reasoning_present,
        "reasoning_separated": reasoning_separated,
        "raw_special_tokens_present": raw_special_tokens_present,
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "result": "PASS" if passed else "FAIL",
        "classification": classification,
    }


def _semantic_assertions(
    body: dict[str, Any] | None,
    *,
    manifest: dict[str, Any],
    unit_text: str,
) -> dict[str, Any]:
    _, message = _message(body)
    calls = message.get("tool_calls") or []
    if len(calls) != 1:
        return {"passed": False, "failures": ["NO_TOOL_CALL"]}
    try:
        arguments = json.loads(calls[0]["function"]["arguments"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return {"passed": False, "failures": ["SCHEMA_INVALID_RESPONSE"]}
    entities = arguments.get("entities") or []
    relations = arguments.get("relations") or []
    temporal_bindings = {
        item.get("relation_local_id", item.get("relation_index")): item.get("surface")
        for item in arguments.get("temporal_bindings") or []
        if isinstance(item, dict)
    }
    compact = bool(entities) and isinstance(entities[0], dict) and (
        "canonical_name" in entities[0]
    )
    names = (
        {index: item.get("canonical_name") for index, item in enumerate(entities)}
        if compact else
        {
            item.get("local_id"): item.get("name")
            for item in entities if isinstance(item, dict)
        }
    )
    keys = {
        (
            names.get(item.get("source_entity_index") if compact else item.get("source_local_id")),
            item.get("raw_relation"),
            names.get(item.get("target_entity_index") if compact else item.get("target_local_id")),
            item.get("negated", (item.get("qualifiers") or {}).get("negated", False)),
        )
        for item in relations if isinstance(item, dict)
    }
    failures: list[str] = []
    for expected in manifest.get("required_relations", []):
        if (
            expected["source"] not in unit_text
            or expected["target"] not in unit_text
        ):
            continue
        key = (
            expected["source"], expected["raw_relation"], expected["target"],
            expected.get("negated", False),
        )
        if key not in keys:
            failures.append(f"MISSING_REQUIRED_RELATION:{key}")
    for forbidden in manifest.get("forbidden_relations", []):
        if (
            forbidden["source"] not in unit_text
            or forbidden["target"] not in unit_text
        ):
            continue
        if any(
            source == forbidden["source"]
            and relation == forbidden["raw_relation"]
            and target == forbidden["target"]
            for source, relation, target, _ in keys
        ):
            failures.append(
                "FORBIDDEN_CONTEXT_ENDPOINT:"
                f"{forbidden['source']}|{forbidden['raw_relation']}|"
                f"{forbidden['target']}"
            )
    for case in manifest.get("required_qualifier_cases", []):
        if case["source"] not in unit_text:
            continue
        matches = [
            {
                **(item.get("qualifiers") or {}),
                "temporality": item.get(
                    "temporal",
                    temporal_bindings.get(
                        index if compact else item.get("relation_local_id"),
                        (item.get("qualifiers") or {}).get("temporality"),
                    ),
                ),
            }
            for index, item in enumerate(relations)
            if names.get(
                item.get("source_entity_index")
                if compact else item.get("source_local_id")
            ) == case["source"]
            and item.get("raw_relation") == case["raw_relation"]
        ]
        if not matches or matches[0].get(case["field"]) != case["value"]:
            failures.append(
                f"QUALIFIER_RECALL_FAILURE:{case['source']}|"
                f"{case['raw_relation']}|{case['field']}"
            )
    return {"passed": not failures, "failures": failures}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--sampling-profile")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config")
    parser.add_argument("--input", type=Path, default=ROOT / "examples" / "sample_input.txt")
    parser.add_argument("--unit-id", default="U005")
    parser.add_argument("--span-id")
    parser.add_argument("--expect-empty-temporal-bindings", action="store_true")
    parser.add_argument("--production-repeats", type=int, default=0)
    parser.add_argument(
        "--production-only",
        action="store_true",
        help="Skip generic capability probes and issue only production requests.",
    )
    parser.add_argument(
        "--capability-tool-only",
        action="store_true",
        help="Issue only the profile-selected minimal tool capability request.",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--semantic-manifest",
        type=Path,
        help="Apply development semantic assertions relevant to the selected unit.",
    )
    args = parser.parse_args(argv)
    if args.production_repeats < 0:
        parser.error("--production-repeats must be non-negative")
    if args.production_only and args.production_repeats == 0:
        parser.error("--production-only requires --production-repeats")
    if args.production_only and args.capability_tool_only:
        parser.error("capability-tool-only cannot be combined with production-only")

    overrides = {"runtime.remote.llm_provider_profile": args.profile}
    if args.sampling_profile:
        overrides[
            f"runtime.remote.provider_profiles.{args.profile}.sampling_profile"
        ] = args.sampling_profile
    settings = load_project_settings(args.config_dir, overrides=overrides)
    remote = settings.runtime.remote
    profile = remote.selected_provider_profile
    probe = remote.capability_probe
    base_url = environment_value(remote.llm_base_url_env).rstrip("/")
    headers = {
        "Authorization": f"Bearer {environment_value(remote.llm_api_key_env)}"
    }
    url = f"{base_url}/chat/completions"

    output: dict[str, Any] = {
        "profile_id": profile.profile_id,
        "model_id": profile.model_id,
        "sampling_profile": args.sampling_profile or profile.sampling_profile,
        "sampling": {
            "temperature": profile.temperature,
            "top_p": profile.top_p,
            "seed": profile.seed,
        },
        "structured_output_transport": profile.structured_output_transport,
        "function_strict": bool(
            payload_strict.function_strict
            if (
                payload_strict := getattr(
                    settings.extraction_contracts[
                        settings.extraction.contract_id
                    ],
                    "structured_output",
                    None,
                )
            ) else False
        ),
        "request_extra_body": profile.request_extra_body,
        "reasoning_policy": profile.reasoning_policy,
        "reasoning_persistence": profile.reasoning_persistence,
        "server_expectations": (
            profile.server_expectations.model_dump()
            if profile.server_expectations else None
        ),
        "plain_chat": None,
        "tool_matrix": {},
        "production_schema_repetitions": [],
    }

    if not args.production_only and not args.capability_tool_only:
        capability_tokens = (
            probe.thinking_max_output_tokens
            if profile.reasoning_policy == "allow_nonempty"
            else probe.plain_chat_max_output_tokens
        )
        plain_payload = _profile_payload(
            profile,
            messages=[{"role": "user", "content": "Reply briefly with OK."}],
            max_tokens=capability_tokens,
        )
        transport, body = execute_probe(
            post=httpx.post, url=url, headers=headers, payload=plain_payload,
            timeout_seconds=probe.timeout_seconds, trust_env=remote.trust_env,
        )
        output["plain_chat"] = _plain_result(transport, body, profile)
        if output["plain_chat"]["result"] != "PASS":
            output["status"] = "blocked"
            output["classification"] = output["plain_chat"]["classification"]
            return _finish(output, args.json_output, 1)

    tool = {
        "type": "function",
        "function": {
            "name": "return_value",
            "description": "Return the requested value.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }
    tool_cases = () if args.production_only else (
        (("auto", "auto"),) if args.capability_tool_only else (
            ("auto", "auto"),
            ("required", "required"),
            ("named", {"type": "function", "function": {"name": "return_value"}}),
            ("none", None),
        )
    )
    for mode, choice in tool_cases:
        payload = _profile_payload(
            profile,
            messages=[
                {"role": "system", "content": "Call return_value exactly once. Return no prose."},
                {"role": "user", "content": "Return the value OK."},
            ],
            max_tokens=(
                probe.thinking_max_output_tokens
                if profile.reasoning_policy == "allow_nonempty"
                else probe.minimal_tool_max_output_tokens
            ),
        )
        payload["tools"] = [tool]
        if choice is not None:
            payload["tool_choice"] = choice
        if profile.parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = profile.parallel_tool_calls
        transport, body = execute_probe(
            post=httpx.post, url=url, headers=headers, payload=payload,
            timeout_seconds=probe.timeout_seconds, trust_env=remote.trust_env,
        )
        output["tool_matrix"][mode] = _minimal_tool_result(
            transport, body, profile
        )

    selected_mode = (
        profile.tool_choice
        if isinstance(profile.tool_choice, str)
        else "named"
        if isinstance(profile.tool_choice, dict)
        else "none"
    )
    selected = (
        {"result": "PASS"} if args.production_only
        else output["tool_matrix"].get(selected_mode)
    )
    if not selected or selected["result"] != "PASS":
        output["status"] = "blocked"
        output["classification"] = (
            selected or {"classification": "SELECTED_TOOL_MODE_FAILED"}
        )["classification"]
        return _finish(output, args.json_output, 1)

    if args.production_repeats:
        bundle = resolve_prompt(settings.extraction.prompt_id)
        dynamic = settings.extraction.dynamic
        schema = bundle.schema_builder(
            max_entities=dynamic.max_entities_per_unit,
            max_relations=dynamic.max_relations_per_unit,
            max_evidence_spans=dynamic.max_evidence_spans_per_item,
            max_relation_description_characters=dynamic.max_relation_description_characters,
        )
        units = split_unit_spans(
            args.input.read_text(encoding="utf-8"),
            strategy=settings.extraction.units.strategy,
        )
        index = int(args.unit_id.removeprefix("U")) - 1
        unit = units[index]
        semantic_manifest = (
            json.loads(args.semantic_manifest.read_text(encoding="utf-8"))
            if args.semantic_manifest else None
        )
        span_cursor = 1 + sum(
            len(sentence_spans(item.text)) for item in units[:index]
        )
        spans = sentence_spans(
            unit.text,
            start_index=span_cursor,
            document_start_offset=unit.start_offset,
        )
        request_unit_id = args.unit_id
        request_text = unit.text
        if args.span_id:
            matching = [item for item in spans if item["span_id"] == args.span_id]
            if len(matching) != 1:
                parser.error(
                    f"--span-id {args.span_id} is not inside {args.unit_id}"
                )
            spans = matching
            request_text = spans[0]["text"]
            request_unit_id = f"{args.unit_id}_{args.span_id}"
        prompt = render_prompt(
            bundle, request_text, spans, unit_id=request_unit_id
        )
        payload = request_payload(settings, prompt, schema)
        payload["max_tokens"] = (
            profile.max_output_tokens
            if profile.reasoning_policy == "allow_nonempty"
            else probe.production_schema_max_output_tokens
        )
        sampling_independent_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"temperature", "top_p", "seed"}
        }
        output.update({
            "prompt_id": bundle.prompt_id,
            "prompt_hash": bundle.content_hash,
            "tool_schema_hash": hashlib.sha256(
                json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "unit_id": request_unit_id,
            "span_id": args.span_id,
            "sampling_independent_payload_hash": hashlib.sha256(
                json.dumps(
                    sampling_independent_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "request_fingerprint": {
                "system_prompt_hash": hashlib.sha256(
                    payload["messages"][0]["content"].encode()
                ).hexdigest(),
                "user_payload_hash": hashlib.sha256(
                    payload["messages"][1]["content"].encode()
                ).hexdigest(),
                "tool_schema_hash": hashlib.sha256(
                    json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "provider_profile_id": profile.profile_id,
                "sampling_profile_id": profile.sampling_profile,
                "tool_choice": payload.get("tool_choice"),
                "max_output_tokens": payload.get("max_tokens"),
                "temperature": payload.get("temperature"),
                "top_p_present": "top_p" in payload,
                "extra_body_hash": hashlib.sha256(
                    json.dumps(
                        profile.request_extra_body,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            },
        })
        for _ in range(args.production_repeats):
            transport, body = execute_probe(
                post=httpx.post, url=url, headers=headers, payload=payload,
                timeout_seconds=probe.timeout_seconds, trust_env=remote.trust_env,
            )
            row = evaluate_structured_response(
                transport=transport,
                body=body,
                expected_function="submit_dynamic_inventory",
                contract=bundle.contract,
                valid_span_ids={span["span_id"] for span in spans},
                reasoning_response_fields=profile.reasoning_response_fields,
                reasoning_policy=profile.reasoning_policy,
                accepted_finish_reasons=profile.accepted_finish_reasons,
            )
            if row["pydantic_valid"] and isinstance(
                bundle.contract, type
            ) and issubclass(bundle.contract, SpanSemanticTemporalExtraction):
                try:
                    _, message = _message(body)
                    arguments = json.loads(
                        message["tool_calls"][0]["function"]["arguments"]
                    )
                    contract_output = bundle.contract.model_validate(arguments)
                    enriched = validate_and_enrich_semantic_temporal(
                        contract_output,
                        unit_id=request_unit_id,
                        span_id=spans[0]["span_id"],
                        evidence_text=spans[0]["text"],
                    )
                    row["span_semantic_temporal_contract"] = {
                        "passed": True,
                        "failure_code": None,
                        "relation_count": len(contract_output.relations),
                        "relation_family_present": all(
                            item.relation_family is not None
                            for item in contract_output.relations
                        ),
                        "binding_count": len(contract_output.temporal_bindings),
                        "binding_relation_indices": [
                            item.relation_index
                            for item in contract_output.temporal_bindings
                        ],
                        "surfaces": [
                            item.surface for item in contract_output.temporal_bindings
                        ],
                        "adapter_valid": bool(enriched),
                    }
                    if (
                        args.expect_empty_temporal_bindings
                        and contract_output.temporal_bindings
                    ):
                        row["span_semantic_temporal_contract"].update({
                            "passed": False,
                            "failure_code": "UNEXPECTED_TEMPORAL_BINDING",
                        })
                        row["result"] = "FAIL"
                except ExtractionContractViolation as error:
                    row["span_semantic_temporal_contract"] = {
                        "passed": False, "failure_code": error.code,
                    }
                    row["result"] = "FAIL"
            if semantic_manifest is not None:
                semantic = _semantic_assertions(
                    body, manifest=semantic_manifest, unit_text=request_text,
                )
                row["semantic_assertions"] = {
                    **semantic,
                    "blocking": False,
                }
            output["production_schema_repetitions"].append(row)
        if not all(
            item["result"] == "PASS"
            for item in output["production_schema_repetitions"]
        ):
            output["status"] = "failed"
            output["classification"] = "PRODUCTION_SCHEMA_QUALIFICATION_FAILED"
            return _finish(output, args.json_output, 1)

    output["status"] = "passed"
    output["classification"] = None
    return _finish(output, args.json_output, 0)


def _finish(
    output: dict[str, Any], path: Path | None, returncode: int,
) -> int:
    serialized = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
