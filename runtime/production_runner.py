"""Supported single-pass fact-extraction production orchestration."""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import httpx
from pydantic import ValidationError

from config.settings import environment_value, load_project_settings
from prompts.contracts import FactExtraction, SpanSemanticTemporalExtraction
from prompts.registry import PromptBundle, resolve_prompt
from runtime.artifact_store import RunArtifacts
from runtime.final_graph_contract import validate_final_graph
from runtime.graph_view import GraphRun, write_graph_view
from runtime.json_neo4j_parity import import_neo4j
from runtime.projection_manifest import projection_manifest
from runtime.model_call_budget import ProductionModelCallBudget
from runtime.observable_entity_resolution import resolve_entities
from runtime.production_inputs import (
    effective_request_identity, render_prompt, request_payload, sentence_spans,
    split_unit_spans,
)
from runtime.production_projection import project_extraction_unit
from runtime.run_reports import deduplicate_exact_edges, run_report, validation_report
from runtime.run_identity import (
    build_run_identity, generate_run_id, source_identity, validate_run_id,
)
from runtime.transport_recovery import parse_tool_output
from runtime.provider_compatibility import discard_reasoning_text, reasoning_metadata
from runtime.span_extraction import aggregate_namespaced_results
from runtime.span_semantic_temporal_contract import (
    validate_and_enrich_semantic_temporal,
)

ROOT = Path(__file__).resolve().parents[1]


def globals_from(entities: list[dict[str, Any]], ids: dict[str, str]) -> list[dict[str, Any]]:
    groups: dict[str,list[dict[str,Any]]] = {}
    for entity in entities: groups.setdefault(ids[entity["local_id"]], []).append(entity)
    return [{"entity_id":entity_id,"canonical_name":rows[0]["name"],"primary_type":rows[0]["type"],
             "aliases":sorted({row["name"] for row in rows[1:]}),"local_entity_ids":[row["local_id"] for row in rows],
             "evidence_span_ids":sorted({span for row in rows for span in row["evidence_span_ids"]})}
            for entity_id,rows in groups.items()]


def _save_budget(artifacts: RunArtifacts, budget: ProductionModelCallBudget) -> None:
    artifacts.json("model_call_attempts.json", budget.as_dict())


def validate_extraction_evidence(
    *,
    local_entities: list[dict[str, Any]],
    local_relations: list[dict[str, Any]],
    valid_span_ids: set[str],
) -> None:
    unknown = sorted({
        span
        for item in [*local_entities, *local_relations]
        for span in item["evidence_span_ids"]
        if span not in valid_span_ids
    })
    if unknown:
        raise RuntimeError(
            f"extraction references unknown evidence span IDs: {unknown}"
        )


def _extract_span_units(
    *, unit_texts: list[str], unit_document_offsets: list[int], settings: Any,
    prompt_bundle: PromptBundle, schema: dict[str, Any], artifacts: RunArtifacts,
    post: Callable[..., Any], budget: ProductionModelCallBudget,
) -> tuple[list[tuple[str,FactExtraction]],list[dict[str,Any]],list[dict[str,Any]],list[str],list[dict[str,str]],list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]]]:
    tasks: list[tuple[int, str, dict[str, Any]]] = []
    all_spans: list[dict[str, Any]] = []
    span_cursor = 1
    for parent_index, (unit_text, document_offset) in enumerate(
        zip(unit_texts, unit_document_offsets, strict=True), start=1
    ):
        spans = sentence_spans(
            unit_text,
            start_index=span_cursor,
            document_start_offset=document_offset,
        )
        span_cursor += len(spans)
        for span in spans:
            request_unit_id = f"U{parent_index:03d}_{span['span_id']}"
            tasks.append((len(tasks), request_unit_id, span))
            all_spans.append(span)

    lock = threading.Lock()
    remote = settings.runtime.remote
    profile = remote.selected_provider_profile

    def worker(task: tuple[int, str, dict[str, Any]]) -> tuple[Any, ...]:
        order, request_unit_id, span = task
        prompt = render_prompt(
            prompt_bundle,
            span["text"],
            [span],
            unit_id=request_unit_id,
        )
        payload = request_payload(settings, prompt, schema)
        request_path = artifacts.request(request_unit_id)
        response_path = artifacts.response(request_unit_id)
        artifacts.json(str(request_path.relative_to(artifacts.root)), payload)
        started = time.perf_counter()
        with lock:
            budget.unit(request_unit_id).record_http_attempt()
            _save_budget(artifacts, budget)
        try:
            response = post(
                f"{environment_value(remote.llm_base_url_env).rstrip('/')}/chat/completions",
                headers={
                    "Authorization": (
                        f"Bearer {environment_value(remote.llm_api_key_env)}"
                    )
                },
                json=payload,
                timeout=profile.timeout_seconds,
                trust_env=remote.trust_env,
            )
            try:
                body = response.json() if response.content else {}
            except ValueError:
                body = {"non_json_body": response.text}
            persisted_body, reasoning = discard_reasoning_text(
                body, profile.reasoning_response_fields,
            )
            artifacts.json(
                str(response_path.relative_to(artifacts.root)),
                {
                    "http_status": response.status_code,
                    "body": persisted_body,
                    "latency_seconds": round(time.perf_counter() - started, 3),
                    **reasoning,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            with lock:
                budget.unit(request_unit_id).record_transport_failure(
                    timed_out=isinstance(error, httpx.TimeoutException),
                    subcategory=(
                        "timeout" if isinstance(error, httpx.TimeoutException)
                        else "http_error"
                    ),
                )
                _save_budget(artifacts, budget)
            raise
        usage = body.get("usage") or {}
        with lock:
            budget.unit(request_unit_id).record_transport_success(
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                reasoning_present=reasoning_metadata(
                    body, profile.reasoning_response_fields,
                )["reasoning_present"],
            )
            _save_budget(artifacts, budget)
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            with lock:
                budget.unit(request_unit_id).record_rejection(
                    "provider_response_rejected", "missing_choices",
                )
                _save_budget(artifacts, budget)
            raise RuntimeError("provider response is missing a usable choice")
        choice = choices[0]
        if choice.get("finish_reason") not in profile.accepted_finish_reasons:
            with lock:
                budget.unit(request_unit_id).record_rejection(
                    "provider_response_rejected", "invalid_finish_reason",
                )
                _save_budget(artifacts, budget)
            raise RuntimeError("provider returned an unsupported finish_reason")
        message = choice.get("message") or {}
        if profile.reasoning_policy == "require_empty" and any(
            message.get(field) for field in profile.reasoning_response_fields
        ):
            with lock:
                budget.unit(request_unit_id).record_rejection(
                    "provider_response_rejected", "reasoning_policy_violation",
                )
                _save_budget(artifacts, budget)
            raise RuntimeError(
                "provider returned reasoning despite require_empty policy"
            )
        try:
            parsed, transport = parse_tool_output(
                message, transport=profile.structured_output_transport,
            )
        except (RuntimeError, ValueError, TypeError, json.JSONDecodeError) as error:
            message_text = str(error)
            subcategory = (
                "wrong_tool_name" if "wrong function" in message_text
                else "malformed_arguments" if isinstance(error, json.JSONDecodeError)
                else "tool_contract_violation"
            )
            with lock:
                budget.unit(request_unit_id).record_rejection(
                    "tool_contract_failed", subcategory,
                )
                _save_budget(artifacts, budget)
            raise
        artifacts.json(
            str(artifacts.parsed(request_unit_id).relative_to(artifacts.root)),
            parsed,
        )
        try:
            contract_output = prompt_bundle.contract.model_validate(parsed)
        except ValidationError:
            with lock:
                budget.unit(request_unit_id).record_rejection(
                    "schema_failed", "pydantic_validation_failed",
                )
                _save_budget(artifacts, budget)
            raise
        with lock:
            budget.unit(request_unit_id).record_schema_valid()
            _save_budget(artifacts, budget)
        if not isinstance(contract_output, SpanSemanticTemporalExtraction):
            raise TypeError("registered extraction contract has an unexpected type")
        try:
            namespaced = validate_and_enrich_semantic_temporal(
                contract_output,
                unit_id=request_unit_id,
                span_id=span["span_id"],
                evidence_text=span["text"],
            )
        except (RuntimeError, ValueError, TypeError):
            with lock:
                budget.unit(request_unit_id).record_rejection(
                    "evidence_failed", "temporal_evidence_failed",
                )
                _save_budget(artifacts, budget)
            raise
        with lock:
            budget.unit(request_unit_id).record_evidence_valid()
            budget.unit(request_unit_id).record_accepted()
            _save_budget(artifacts, budget)
        return order, namespaced, choice, usage, transport

    completed: list[tuple[Any, ...]] = []
    failures: list[tuple[str, BaseException]] = []
    with ThreadPoolExecutor(
        max_workers=settings.extraction.max_concurrency
    ) as executor:
        future_units = {
            executor.submit(worker, task): task[1] for task in tasks
        }
        for future in as_completed(future_units):
            try:
                completed.append(future.result())
            except BaseException as error:
                failures.append((future_units[future], error))
    if failures:
        artifacts.json("terminal_failure.json", {
            "status": "terminal_validation_failure",
            "rule_category": "span_flat_extraction",
            "failure_policy": settings.extraction.failure_policy,
            "failed_units": [unit_id for unit_id, _ in failures],
            "errors": [f"{type(error).__name__}: {error}" for _, error in failures],
        })
        raise RuntimeError(
            f"span-flat extraction failed for {len(failures)} span units"
        ) from failures[0][1]
    completed.sort(key=lambda item: item[0])
    namespaced = [item[1] for item in completed]
    artifacts.json("span_extraction_results.json", namespaced)
    with lock:
        for item in namespaced:
            budget.unit(item["unit_id"]).record_persisted_span()
        _save_budget(artifacts, budget)
    extractions = aggregate_namespaced_results(namespaced)
    choices = [item[2] for item in completed]
    usages = [item[3] for item in completed]
    transports = [item[4] for item in completed]
    unit_records = [
        {"unit_id": item["unit_id"], "span_ids": [item["span_id"]]}
        for item in namespaced
    ]
    local_entities = [
        entity.model_dump(mode="json") | {"unit_id": unit_id}
        for unit_id, extraction in extractions for entity in extraction.entities
    ]
    local_relations = [
        relation.model_dump(mode="json") | {"unit_id": unit_id}
        for unit_id, extraction in extractions
        for relation in extraction.relations
    ]
    return (
        extractions, choices, usages, transports, all_spans, unit_records,
        local_entities, local_relations,
    )


def run(input_path: Path, output_dir: Path, config_dir: Path, *, run_id: str|None=None, candidate_threshold: float|None=None,
        merge_threshold: float|None=None, llm_post: Callable[...,Any]=httpx.post,
        embedding_resolver: Callable[...,Any]=resolve_entities, neo4j_importer: Callable[...,Any]=import_neo4j) -> int:
    overrides={}
    if candidate_threshold is not None: overrides["entity_resolution.thresholds.candidate_similarity"]=candidate_threshold
    if merge_threshold is not None: overrides["entity_resolution.thresholds.automatic_merge_similarity"]=merge_threshold
    if output_dir.exists():
        raise FileExistsError(
            "The output directory already exists. Resume is not supported. "
            "Choose a new output directory or remove the incomplete run explicitly."
        )
    settings=load_project_settings(config_dir,overrides=overrides); prompt_bundle=resolve_prompt(settings.extraction.prompt_id); resolved_config=settings.resolved_for_artifact(); source=input_path.read_text(encoding="utf-8"); source_metadata=source_identity(input_path,source,ROOT); unit_spans=split_unit_spans(source,strategy=settings.extraction.units.strategy); unit_texts=[unit.text for unit in unit_spans]; artifacts=RunArtifacts(output_dir)
    prompt_identity={"prompt_id":prompt_bundle.prompt_id,"prompt_hash":prompt_bundle.content_hash}
    dynamic=settings.extraction.dynamic
    schema=prompt_bundle.schema_builder(max_entities=dynamic.max_entities_per_unit,max_relations=dynamic.max_relations_per_unit,max_evidence_spans=dynamic.max_evidence_spans_per_item,max_relation_description_characters=dynamic.max_relation_description_characters)
    request_identity=effective_request_identity(settings,prompt_bundle,schema)
    resolved_run_id=validate_run_id(run_id) if run_id is not None else generate_run_id()
    contract_id=settings.extraction.contract_id; contract_settings=settings.extraction_contracts[contract_id]; schema_id=contract_settings.schema_id
    run_identity=build_run_identity(run_id=resolved_run_id,source=source_metadata,prompt_hash=prompt_bundle.content_hash,schema_hash=request_identity["schema_hash"],contract_id=contract_id,provider_profile_id=request_identity["provider_profile_id"])
    graph_id=run_identity["graph_id"]
    output_dir.mkdir(parents=True,exist_ok=True); artifacts.json("resolved_run_config.json",resolved_config); artifacts.json("prompt_manifest.json",prompt_identity|{"contract_id":contract_id,"schema_id":schema_id,"request_identity":request_identity}); artifacts.json("run_manifest.json",run_identity|{"source":source_metadata}); artifacts.text("source.txt",source)
    budget=ProductionModelCallBudget()
    extraction_arguments = {
        "unit_texts": unit_texts,
        "unit_document_offsets": [unit.start_offset for unit in unit_spans],
        "settings": settings,
        "prompt_bundle": prompt_bundle,
        "schema": schema,
        "artifacts": artifacts,
        "post": llm_post,
        "budget": budget,
    }
    extractions,choices,usages,transports,all_spans,unit_records,local_entities,local_relations=_extract_span_units(
        **extraction_arguments
    )
    artifacts.json("spans.json",all_spans); artifacts.json("local_entities.json",local_entities); artifacts.json("local_relations.json",local_relations)
    valid_spans={item["span_id"] for item in all_spans}
    try:
        validate_extraction_evidence(
            local_entities=local_entities,
            local_relations=local_relations,
            valid_span_ids=valid_spans,
        )
    except RuntimeError as error:
        artifacts.json("terminal_failure.json", {
            "status": "terminal_validation_failure",
            "rule_category": "extraction_evidence",
            "error": str(error),
        })
        raise
    ers=settings.entity_resolution; remote=settings.runtime.remote
    resolution,resolution_report=embedding_resolver(local_entities,local_relations,endpoint=environment_value(remote.embedding_base_url_env),api_key=environment_value(remote.embedding_api_key_env),model=ers.embedding.model,candidate_threshold=ers.thresholds.candidate_similarity,merge_threshold=ers.thresholds.automatic_merge_similarity,decision_settings=ers.decisions.model_dump(),pairwise_limit=ers.retrieval.small_run_pairwise_limit,top_k=ers.retrieval.top_k,timeout_seconds=remote.embedding_timeout_seconds,trust_env=remote.trust_env,write_artifact=artifacts.json)
    entities=globals_from(local_entities,resolution); artifacts.json("entity_resolution.json",{"local_to_global":resolution,**resolution_report}); artifacts.json("global_entities.json",entities)
    edges=[]
    for unit_id,extraction in extractions: edges.extend(project_extraction_unit(unit_id=unit_id,extraction=extraction,resolved_entity_ids=resolution,valid_span_ids=valid_spans,prompt_identity=prompt_identity)["edges"])
    edges,duplicates=deduplicate_exact_edges(edges)
    candidate_graph={"graph_id":graph_id,"run_id":resolved_run_id,"manifest_identity_hash":run_identity["manifest_identity_hash"],**prompt_identity,"nodes":entities,"edges":edges,
                     "source":source_metadata|{"units":unit_records}}
    try:
        graph=validate_final_graph(candidate_graph).model_dump(mode="json")
    except ValidationError as error:
        artifacts.json("terminal_failure.json",{
            "status":"terminal_validation_failure",
            "rule_category":"final_graph_contract",
            "errors":error.errors(
                include_url=False,include_input=False,include_context=False
            ),
        })
        raise RuntimeError("final graph contract validation failed before persistence") from error
    artifacts.json("fact_graph.json",graph)
    if budget.as_dict()["totals"]["embedding_calls"] == 0:
        budget.unit(extractions[0][0]).record_embedding(int(resolution_report.get("embedder_calls") or 0)); _save_budget(artifacts,budget)
    budget.assert_valid(); budget_payload=budget.as_dict(); artifacts.json("model_call_budget.json",budget_payload)
    ownership={key:run_identity[key] for key in ("run_id","source_sha256","manifest_identity_hash")}
    artifacts.json("projection_manifest.json", projection_manifest(
        graph_id=graph_id,
        projection=settings.graph_projection.type_names,
    ))
    neo4j,parity=neo4j_importer(graph_id,entities,edges,settings,ownership); parity["duplicate_edges"]=[item["edge_id"] for item in duplicates]; artifacts.json("json_neo4j_edge_diff.json",parity)
    validation=validation_report(choices=choices,transports=transports,local_relations=local_relations,valid_spans=valid_spans,budget=budget_payload,parity=parity); artifacts.json("validation_report.json",validation)
    report_units = [span["text"] for span in all_spans]
    artifacts.json("run_report.json",run_report(run_id=resolved_run_id,graph_id=graph_id,source=source,units=report_units,choices=choices,budget=budget_payload,resolution=resolution_report,local_entities=local_entities,local_relations=local_relations,entities=entities,edges=edges,neo4j=neo4j,validation=validation,transports=transports))
    if not validation["passed"]:
        raise RuntimeError("production validation failed after projection")
    artifacts.json("completion_status.json", {
        "run_status": "completed",
        "extraction_status": "passed",
        "schema_status": "passed",
        "evidence_status": "passed",
        "final_graph_status": "passed",
        "projection_status": "passed",
        "parity_status": "passed",
    })
    write_graph_view(output_dir,GraphRun(resolved_run_id,graph_id,output_dir,len(entities),len(edges)),settings)
    return 0
