"""Sanitized OpenAI-compatible structured-output probe helpers."""
from __future__ import annotations

import json
import hashlib
import time
from typing import Any, Callable

import httpx
from pydantic import BaseModel, ValidationError


def reasoning_metadata(
    body: dict[str, Any], reasoning_response_fields: list[str],
) -> dict[str, Any]:
    """Return safe reasoning metrics without copying reasoning text."""
    choices = body.get("choices") or []
    message = choices[0].get("message") or {} if choices else {}
    present = any(bool(message.get(field)) for field in reasoning_response_fields)
    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    count = details.get("reasoning_tokens")
    return {
        "reasoning_present": present,
        "reasoning_tokens": int(count) if isinstance(count, int) else None,
    }


def discard_reasoning_text(
    body: dict[str, Any], reasoning_response_fields: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy a response and remove configured reasoning fields before persistence."""
    sanitized = json.loads(json.dumps(body))
    metadata = reasoning_metadata(sanitized, reasoning_response_fields)
    for choice in sanitized.get("choices") or []:
        message = choice.get("message") if isinstance(choice, dict) else None
        if isinstance(message, dict):
            for field in reasoning_response_fields:
                message.pop(field, None)
    return sanitized, metadata


def apply_thinking_override(
    payload: dict[str, Any], mode: str,
) -> dict[str, Any]:
    """Return a copied payload with one explicit provider compatibility option."""
    result = json.loads(json.dumps(payload))
    if mode == "none":
        return result
    if mode == "chat_template_kwargs":
        result["chat_template_kwargs"] = {"enable_thinking": False}
    elif mode == "enable_thinking":
        result["enable_thinking"] = False
    elif mode == "reasoning":
        result["reasoning"] = False
    else:
        raise ValueError(f"unsupported thinking override: {mode}")
    return result


def execute_probe(
    *,
    post: Callable[..., Any],
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_seconds: int,
    trust_env: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Execute one request and return sanitized transport metadata plus JSON body."""
    started = time.perf_counter()
    try:
        response = post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout_seconds,
            trust_env=trust_env,
        )
        duration = round(time.perf_counter() - started, 3)
        try:
            body = response.json() if response.content else {}
        except ValueError:
            body = None
        metadata = {
            "http_status": response.status_code,
            "duration_seconds": duration,
            "response_json": isinstance(body, dict),
            "exception_type": None,
        }
        return metadata, body if isinstance(body, dict) else None
    except httpx.HTTPError as error:
        return {
            "http_status": None,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "response_json": False,
            "exception_type": type(error).__name__,
        }, None


def evaluate_structured_response(
    *,
    transport: dict[str, Any],
    body: dict[str, Any] | None,
    expected_function: str,
    contract: type[BaseModel],
    valid_span_ids: set[str],
    reasoning_response_fields: list[str] | None = None,
    reasoning_policy: str = "require_empty",
    accepted_finish_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Summarize tool-call, schema, and evidence validity without response content."""
    result = dict(transport)
    result.update({
        "finish_reason": None,
        "tool_calls": 0,
        "correct_function": False,
        "arguments_json_object": False,
        "pydantic_valid": False,
        "evidence_valid": False,
        "reasoning_absent": True,
        "reasoning_present": False,
        "reasoning_tokens": None,
        "reasoning_separated": False,
        "raw_special_tokens_present": False,
        "truncated": False,
        "content_and_tool_coexist": False,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "result": "FAIL",
        "schema_failure_categories": [],
        "schema_issues": [],
        "structure": None,
        "response_argument_hash": None,
        "entity_count": None,
        "relation_count": None,
        "relation_family_multiset": [],
        "raw_relation_normalized_hashes": [],
    })
    if not body:
        return result
    choices = body.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return result
    choice = choices[0]
    message = choice.get("message") or {}
    calls = message.get("tool_calls") or []
    result["finish_reason"] = choice.get("finish_reason")
    result["tool_calls"] = len(calls)
    result["content_and_tool_coexist"] = bool(message.get("content")) and bool(calls)
    reasoning_fields = reasoning_response_fields or ["reasoning", "reasoning_content"]
    result["reasoning_absent"] = not any(
        message.get(field) for field in reasoning_fields
    )
    result["reasoning_present"] = not result["reasoning_absent"]
    result["reasoning_separated"] = result["reasoning_present"] and not bool(
        message.get("content")
    )
    content = message.get("content")
    result["raw_special_tokens_present"] = isinstance(content, str) and any(
        token in content for token in ("<think>", "</think>", "<|tool_call_")
    )
    result["truncated"] = choice.get("finish_reason") == "length"
    usage = body.get("usage") or {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        result[key] = int(usage.get(key) or 0)
    details = usage.get("completion_tokens_details") or {}
    if isinstance(details.get("reasoning_tokens"), int):
        result["reasoning_tokens"] = details["reasoning_tokens"]
    if len(calls) != 1 or not isinstance(calls[0], dict):
        return result
    function = calls[0].get("function") or {}
    result["correct_function"] = function.get("name") == expected_function
    try:
        arguments = json.loads(function.get("arguments", ""))
    except (TypeError, json.JSONDecodeError):
        return result
    result["arguments_json_object"] = isinstance(arguments, dict)
    if not isinstance(arguments, dict):
        return result
    canonical_arguments = json.dumps(
        arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    result["response_argument_hash"] = hashlib.sha256(
        canonical_arguments.encode()
    ).hexdigest()
    entities = arguments.get("entities")
    relations = arguments.get("relations")
    relation_rows = relations if isinstance(relations, list) else []
    result["entity_count"] = len(entities) if isinstance(entities, list) else None
    result["relation_count"] = len(relation_rows) if isinstance(relations, list) else None
    result["relation_family_multiset"] = sorted(
        str(item.get("relation_family"))
        for item in relation_rows
        if isinstance(item, dict) and item.get("relation_family") is not None
    )
    result["raw_relation_normalized_hashes"] = [
        hashlib.sha256(
            " ".join(str(item.get("raw_relation", "")).casefold().split()).encode()
        ).hexdigest()
        for item in relation_rows if isinstance(item, dict)
    ]
    result["structure"] = {
        "entities_is_array": isinstance(entities, list),
        "relations_is_array": isinstance(relations, list),
        "temporal_present_by_relation": [
            "temporal" in item if isinstance(item, dict) else False
            for item in relation_rows
        ],
        "temporal_type_by_relation": [
            (
                "null" if item.get("temporal") is None else
                "string" if isinstance(item.get("temporal"), str) else
                type(item.get("temporal")).__name__
            ) if isinstance(item, dict) and "temporal" in item else "absent"
            for item in relation_rows
        ],
    }
    try:
        extraction = contract.model_validate(arguments)
    except ValidationError as error:
        categories: set[str] = set()
        for item in error.errors(include_url=False, include_input=False):
            result["schema_issues"].append({
                "path": ".".join(str(part) for part in item.get("loc", ())),
                "type": item["type"],
            })
            if item["type"] == "missing":
                categories.add("MISSING_REQUIRED_FIELD")
            elif (
                item["type"] == "string_pattern_mismatch"
                and item.get("loc", ())[-1:] in {
                    ("local_id",), ("source_local_id",), ("target_local_id",)
                }
            ):
                categories.add("INVALID_LOCAL_ID")
            else:
                categories.add("SCHEMA_INVALID_RESPONSE")
        result["schema_failure_categories"] = sorted(categories)
        return result
    result["pydantic_valid"] = True
    referenced = {
        span
        for item in [*extraction.entities, *extraction.relations]
        for span in getattr(item, "evidence_span_ids", [])
    }
    result["evidence_valid"] = referenced <= valid_span_ids
    reasoning_valid = (
        result["reasoning_absent"] if reasoning_policy == "require_empty"
        else result["reasoning_present"] and result["reasoning_separated"]
        if reasoning_policy == "allow_nonempty"
        else True
    )
    if (
        result["http_status"] == 200
        and result["correct_function"]
        and result["arguments_json_object"]
        and result["pydantic_valid"]
        and result["evidence_valid"]
        and reasoning_valid
        and not result["raw_special_tokens_present"]
        and not result["truncated"]
        and (
            accepted_finish_reasons is None
            or result["finish_reason"] in accepted_finish_reasons
        )
    ):
        result["result"] = "PASS"
    return result
