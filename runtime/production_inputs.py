"""Deterministic input splitting and registered prompt request construction."""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from typing import Any

from config.settings import ProjectSettings
from prompts.registry import PromptBundle


@dataclass(frozen=True)
class UnitSpan:
    text: str
    start_offset: int
    end_offset: int


def _trimmed_bounds(source: str, start: int, end: int) -> tuple[int, int]:
    while start < end and source[start].isspace():
        start += 1
    while end > start and source[end - 1].isspace():
        end -= 1
    return start, end


def sentence_spans(
    source: str, *, start_index: int = 1, document_start_offset: int = 0,
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    start = 0
    for index, character in enumerate(source):
        if character not in ".!?":
            continue
        next_character = source[index + 1] if index + 1 < len(source) else ""
        if character == "." and next_character and not next_character.isspace():
            continue
        span_start, span_end = _trimmed_bounds(source, start, index + 1)
        if span_start < span_end:
            spans.append({
                "span_id": f"S{start_index + len(spans):03d}",
                "text": source[span_start:span_end],
                "start_offset": span_start,
                "end_offset": span_end,
                "document_start_offset": document_start_offset + span_start,
                "document_end_offset": document_start_offset + span_end,
            })
        start = index + 1
    span_start, span_end = _trimmed_bounds(source, start, len(source))
    if span_start < span_end:
        spans.append({
            "span_id": f"S{start_index + len(spans):03d}",
            "text": source[span_start:span_end],
            "start_offset": span_start,
            "end_offset": span_end,
            "document_start_offset": document_start_offset + span_start,
            "document_end_offset": document_start_offset + span_end,
        })
    return spans


def split_unit_spans(source: str, *, strategy: str) -> list[UnitSpan]:
    if strategy != "paragraph":
        raise ValueError(f"unsupported unit splitting strategy: {strategy}")
    units: list[UnitSpan] = []
    paragraph_start: int | None = None
    paragraph_end = 0
    cursor = 0
    for line in source.splitlines(keepends=True):
        content_end = len(line.rstrip("\r\n"))
        content = line[:content_end]
        if content.strip():
            first = next(index for index, character in enumerate(content) if not character.isspace())
            last = max(index for index, character in enumerate(content) if not character.isspace()) + 1
            if paragraph_start is None:
                paragraph_start = cursor + first
            paragraph_end = cursor + last
        elif paragraph_start is not None:
            units.append(UnitSpan(
                source[paragraph_start:paragraph_end], paragraph_start, paragraph_end,
            ))
            paragraph_start = None
        cursor += len(line)
    if paragraph_start is not None:
        units.append(UnitSpan(source[paragraph_start:paragraph_end], paragraph_start, paragraph_end))
    if not units:
        raise ValueError("input has no extractable units")
    return units


def split_units(source: str, *, strategy: str) -> list[str]:
    return [unit.text for unit in split_unit_spans(source, strategy=strategy)]


def render_prompt(
    bundle: PromptBundle,
    source: str,
    spans: list[dict[str, Any]],
    *,
    unit_id: str = "U001",
) -> str:
    prompt_spans = [
        {"span_id": span["span_id"], "text": span["text"]} for span in spans
    ]
    return bundle.template.format(
        unit_id=unit_id,
        text=source,
        allowed_evidence_spans=json.dumps(
            prompt_spans, ensure_ascii=False, separators=(",", ":")
        ),
        entity_guidance=json.dumps(bundle.ontology["entity_guidance"], ensure_ascii=False, separators=(",", ":")),
        relation_family_guidance=json.dumps(bundle.ontology["relation_families"], ensure_ascii=False, separators=(",", ":")),
    )


def request_payload(settings: ProjectSettings, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
    system = "Call submit_dynamic_inventory only. Return no prose, analysis, or XML."
    remote = settings.runtime.remote
    profile = remote.selected_provider_profile
    if profile.structured_output_transport != "native_tool_call":
        raise ValueError(
            "production extraction currently supports only native_tool_call transport"
        )
    function = {
        "name": "submit_dynamic_inventory",
        "description": (
            "Submit the local entity inventory and its evidence-backed relations."
        ),
        "parameters": schema,
    }
    contract_settings = settings.extraction_contracts[
        settings.extraction.contract_id
    ]
    structured_output = getattr(contract_settings, "structured_output", None)
    if structured_output is not None and structured_output.function_strict:
        function["strict"] = True
    payload = {
        "model": profile.model_id,
        "max_tokens": profile.max_output_tokens,
        "messages": [{"role":"system","content":system},{"role":"user","content":prompt}],
        "tools": [{"type": "function", "function": function}],
    }
    optional = {
        "tool_choice": profile.tool_choice,
        "parallel_tool_calls": profile.parallel_tool_calls,
        "stream": profile.stream,
        "temperature": profile.temperature,
        "top_p": profile.top_p,
        "seed": profile.seed,
    }
    payload.update({key: value for key, value in optional.items() if value is not None})
    payload.update(json.loads(json.dumps(profile.request_extra_body)))
    payload.update(remote.llm_request_overrides)
    return payload


def effective_request_identity(
    settings: ProjectSettings, bundle: PromptBundle, schema: dict[str, Any],
) -> dict[str, Any]:
    """Return a secret-free identity for the effective model request contract."""
    remote = settings.runtime.remote
    profile = remote.selected_provider_profile
    extras = json.loads(json.dumps(profile.request_extra_body))
    extras.update(json.loads(json.dumps(remote.llm_request_overrides)))
    canonical = lambda value: json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return {
        "model_id": profile.model_id,
        "provider_profile_id": profile.profile_id,
        "sampling_profile_id": profile.sampling_profile,
        "prompt_hash": bundle.content_hash,
        "schema_hash": hashlib.sha256(canonical(schema)).hexdigest(),
        "tool_name": "submit_dynamic_inventory",
        "tool_choice": profile.tool_choice,
        "allowed_extra_body_hash": hashlib.sha256(canonical(extras)).hexdigest(),
    }
