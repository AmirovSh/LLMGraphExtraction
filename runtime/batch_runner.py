"""Multi-document orchestration with graph-group isolation."""
from __future__ import annotations

import json
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable

import httpx

from config.settings import environment_value, load_project_settings
from prompts.registry import resolve_prompt
from runtime.artifact_store import RunArtifacts, sha256_json, write_json
from runtime.batch_manifest import ResolvedBatchPlan, resolve_batch_plan
from runtime.batch_scheduler import ExtractionQueueItem, FairRoundRobinScheduler
from runtime.final_graph_contract import validate_final_graph
from runtime.json_neo4j_parity import import_neo4j
from runtime.model_call_budget import ProductionModelCallBudget
from runtime.observable_entity_resolution import resolve_entities
from runtime.production_inputs import sentence_spans, split_unit_spans
from runtime.production_projection import project_extraction_unit
from runtime.production_runner import _extract_span_units, globals_from
from runtime.run_reports import deduplicate_exact_edges
from runtime.span_extraction import aggregate_namespaced_results


def _document_spans(plan: ResolvedBatchPlan, prompt_hash: str, schema_hash: str,
                    provider_profile: str) -> list[ExtractionQueueItem]:
    items: list[ExtractionQueueItem] = []
    for document in plan.documents:
        source = document.path.read_text(encoding="utf-8")
        span_cursor = 1
        order = 0
        for unit in split_unit_spans(source, strategy="paragraph"):
            spans = sentence_spans(unit.text, start_index=span_cursor,
                                   document_start_offset=unit.start_offset)
            span_cursor += len(spans)
            for span in spans:
                order += 1
                items.append(ExtractionQueueItem(
                    batch_id=plan.manifest.batch.id, graph_id=document.graph_id,
                    document_id=document.document_id, span_id=span["span_id"],
                    document_order=order, source_start=span["document_start_offset"],
                    source_end=span["document_end_offset"],
                    estimated_input_tokens=max(1, len(span["text"]) // 4),
                    prompt_hash=prompt_hash, schema_hash=schema_hash,
                    provider_profile=provider_profile, enqueue_timestamp=time.time(),
                    payload=span,
                ))
    return items


def _extract_item(item: ExtractionQueueItem, *, settings: Any, bundle: Any,
                  schema: dict[str, Any], root: Path, post: Callable[..., Any]) -> dict[str, Any]:
    artifacts = RunArtifacts(root / "documents" / item.document_id / "units" /
                             f"span_{item.document_order:06d}")
    budget = ProductionModelCallBudget()
    result = _extract_span_units(
        unit_texts=[item.payload["text"]], unit_document_offsets=[item.source_start],
        settings=settings, prompt_bundle=bundle, schema=schema, artifacts=artifacts,
        post=post, budget=budget,
    )
    extractions, choices, usages, transports, _, _, _, _ = result
    namespaced = json.loads(artifacts.path("span_extraction_results.json").read_text(encoding="utf-8"))[0]
    old_span = namespaced["span_id"]
    evidence_id = f"{item.document_id}:{item.span_id}"
    unit_id = f"{item.document_id}:U{item.document_order:06d}"
    namespaced["unit_id"] = unit_id
    namespaced["span_id"] = evidence_id
    for entity in namespaced["entities"]:
        entity["evidence_span_ids"] = [evidence_id]
        entity["local_id"] = entity["local_id"].replace(old_span, evidence_id, 1)
    for relation in namespaced["relations"]:
        relation["evidence_span_ids"] = [evidence_id]
        for key in ("relation_local_id", "source_local_id", "target_local_id"):
            relation[key] = relation[key].replace(old_span, evidence_id, 1)
    totals = budget.as_dict()["totals"]
    return {
        "item": item, "namespaced": namespaced, "choice": choices[0],
        "usage": usages[0], "transport": transports[0], "budget": totals,
        "unit_id": unit_id, "evidence_id": evidence_id,
    }


def _finalize_graph(graph_plan: Any, results: list[dict[str, Any]], *, plan: ResolvedBatchPlan,
                    settings: Any, bundle: Any, root: Path,
                    embedding_resolver: Callable[..., Any],
                    neo4j_importer: Callable[..., Any]) -> dict[str, Any]:
    graph_root = root / "graphs" / graph_plan.graph_id
    graph_root.mkdir(parents=True, exist_ok=False)
    ordered = sorted(results, key=lambda row: (row["item"].document_id, row["item"].document_order))
    extractions = aggregate_namespaced_results([row["namespaced"] for row in ordered])
    local_entities = [entity.model_dump(mode="json") | {"unit_id": unit_id}
                      for unit_id, extraction in extractions for entity in extraction.entities]
    local_relations = [relation.model_dump(mode="json") | {"unit_id": unit_id}
                       for unit_id, extraction in extractions for relation in extraction.relations]
    valid_spans = {row["evidence_id"] for row in ordered}
    remote = settings.runtime.remote
    ers = settings.entity_resolution
    resolution, resolution_report = embedding_resolver(
        local_entities, local_relations,
        endpoint=environment_value(remote.embedding_base_url_env),
        api_key=environment_value(remote.embedding_api_key_env), model=ers.embedding.model,
        candidate_threshold=ers.thresholds.candidate_similarity,
        merge_threshold=ers.thresholds.automatic_merge_similarity,
        decision_settings=ers.decisions.model_dump(),
        pairwise_limit=ers.retrieval.small_run_pairwise_limit, top_k=ers.retrieval.top_k,
        timeout_seconds=remote.embedding_timeout_seconds, trust_env=remote.trust_env,
        write_artifact=lambda name, value: write_json(graph_root / name, value),
    )
    entities = globals_from(local_entities, resolution)
    prompt_identity = {"prompt_id": bundle.prompt_id, "prompt_hash": bundle.content_hash}
    edges: list[dict[str, Any]] = []
    for unit_id, extraction in extractions:
        edges.extend(project_extraction_unit(
            unit_id=unit_id, extraction=extraction, resolved_entity_ids=resolution,
            valid_span_ids=valid_spans, prompt_identity=prompt_identity,
        )["edges"])
    edges, duplicates = deduplicate_exact_edges(edges)
    document_rows = [item for item in plan.documents if item.document_id in graph_plan.document_ids]
    source_hash = sha256_json([{ "id": row.document_id, "sha256": row.source_sha256} for row in document_rows])
    manifest_hash = sha256_json({
        "batch_id": plan.manifest.batch.id, "graph_id": graph_plan.graph_id,
        "documents": list(graph_plan.document_ids), "sources": source_hash,
        "request_identity": plan.request_identity,
    })
    namespace = f"batch_{plan.manifest.batch.id}_{graph_plan.graph_id}_{manifest_hash[:16]}"
    candidate = {
        "graph_id": namespace, "run_id": plan.manifest.batch.id,
        "manifest_identity_hash": manifest_hash, **prompt_identity,
        "nodes": entities, "edges": edges,
        "source": {
            "source_name": f"graph-group:{graph_plan.graph_id}",
            "source_sha256": source_hash,
            "source_size_bytes": sum(row.source_size_bytes for row in document_rows),
            "source_location_policy": "external", "repository_relative_path": None,
            "units": [{"unit_id": row["unit_id"], "span_ids": [row["evidence_id"]]} for row in ordered],
        },
    }
    graph = validate_final_graph(candidate).model_dump(mode="json")
    write_json(graph_root / "fact_graph.json", graph)
    graph_manifest = {
        "batch_id": plan.manifest.batch.id, "graph_group_id": graph_plan.graph_id,
        "neo4j_namespace": namespace, "graph_manifest_hash": manifest_hash,
        "document_ids": list(graph_plan.document_ids),
        "documents": [{"document_id": row.document_id, "source_filename": row.source_filename,
                       "source_sha256": row.source_sha256} for row in document_rows],
        "provider_profile": plan.manifest.model.provider_profile,
        "requested_model_id": plan.manifest.model.model_id, **prompt_identity,
        "schema_hash": plan.request_identity["schema_hash"],
    }
    write_json(graph_root / "graph_manifest.json", graph_manifest)
    write_json(graph_root / "resolution_report.json", {
        **resolution_report,
        "source_document_ids": list(graph_plan.document_ids),
        "source_span_ids": sorted(valid_spans),
    })
    ownership = {"run_id": plan.manifest.batch.id, "source_sha256": source_hash,
                 "manifest_identity_hash": manifest_hash}
    neo4j, parity = neo4j_importer(namespace, entities, edges, settings, ownership)
    parity["duplicate_edges"] = [item["edge_id"] for item in duplicates]
    write_json(graph_root / "neo4j_parity.json", parity)
    completed = bool(parity.get("passed", parity.get("status") == "passed"))
    if not completed:
        raise RuntimeError("Neo4j parity failed")
    write_json(graph_root / "completion_status.json", {"status": "completed"})
    return {"graph_id": graph_plan.graph_id, "status": "completed",
            "document_ids": list(graph_plan.document_ids), "span_count": len(ordered),
            "node_count": len(entities), "edge_count": len(edges),
            "model_attempts": sum(int(row["budget"]["primary_http_attempts"]) for row in ordered),
            "input_tokens": sum(int(row["budget"]["prompt_tokens"]) for row in ordered),
            "output_tokens": sum(int(row["budget"]["completion_tokens"]) for row in ordered),
            "embedding_call_count": int(resolution_report.get("embedder_calls") or 0),
            "failure_reason": None, "duration_seconds": None,
            "output_path": f"graphs/{graph_plan.graph_id}"}


def run_batch(manifest_path: Path, config_dir: Path, *, output_override: Path | None = None,
              llm_post: Callable[..., Any] = httpx.post,
              embedding_resolver: Callable[..., Any] = resolve_entities,
              neo4j_importer: Callable[..., Any] = import_neo4j) -> int:
    started = time.perf_counter()
    base_settings = load_project_settings(config_dir)
    plan = resolve_batch_plan(manifest_path, base_settings, output_override=output_override)
    settings = load_project_settings(config_dir, overrides={
        "runtime.remote.llm_provider_profile": plan.manifest.model.provider_profile,
    })
    bundle = resolve_prompt(settings.extraction.prompt_id)
    dynamic = settings.extraction.dynamic
    schema = bundle.schema_builder(max_entities=dynamic.max_entities_per_unit,
        max_relations=dynamic.max_relations_per_unit,
        max_evidence_spans=dynamic.max_evidence_spans_per_item,
        max_relation_description_characters=dynamic.max_relation_description_characters)
    items = _document_spans(plan, bundle.content_hash,
                            str(plan.request_identity["schema_hash"]),
                            plan.manifest.model.provider_profile)
    root = plan.output_directory
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "batch_manifest_resolved.json", plan.public_dict())
    for document in plan.documents:
        doc_root = root / "documents" / document.document_id
        doc_root.mkdir(parents=True, exist_ok=False)
        write_json(doc_root / "source_manifest.json", {
            "document_id": document.document_id, "graph_id": document.graph_id,
            "source_filename": document.source_filename,
            "source_sha256": document.source_sha256,
            "source_size_bytes": document.source_size_bytes,
        })
        write_json(doc_root / "units.json", [{
            "batch_id": item.batch_id, "graph_id": item.graph_id,
            "document_id": item.document_id, "span_id": item.span_id,
            "document_local_order": item.document_order,
            "source_start": item.source_start, "source_end": item.source_end,
            "estimated_input_tokens": item.estimated_input_tokens,
            "prompt_hash": item.prompt_hash, "schema_hash": item.schema_hash,
            "provider_profile": item.provider_profile, "priority": item.priority,
            "enqueue_timestamp": item.enqueue_timestamp,
        } for item in items if item.document_id == document.document_id])
    limits = plan.manifest.execution
    scheduler = FairRoundRobinScheduler(items, global_limit=limits.global_max_in_flight,
        per_document_limit=limits.per_document_max_in_flight,
        per_graph_limit=limits.per_graph_max_in_flight)
    results: dict[str, list[dict[str, Any]]] = {graph.graph_id: [] for graph in plan.graphs}
    failures: dict[str, list[str]] = {}
    failed_items: list[ExtractionQueueItem] = []
    active: dict[Future[dict[str, Any]], ExtractionQueueItem] = {}
    stop_new = False
    scheduled_count = 0
    with ThreadPoolExecutor(max_workers=limits.global_max_in_flight) as executor:
        while scheduler.pending or active:
            while not stop_new:
                item = scheduler.acquire()
                if item is None:
                    break
                active[executor.submit(_extract_item, item, settings=settings, bundle=bundle,
                    schema=schema, root=root, post=llm_post)] = item
                scheduled_count += 1
            if not active:
                break
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                item = active.pop(future)
                scheduler.complete(item)
                try:
                    results[item.graph_id].append(future.result())
                except BaseException as error:
                    failed_items.append(item)
                    failures.setdefault(item.graph_id, []).append(f"{type(error).__name__}: {error}")
                    scheduler.discard_graph(item.graph_id)
                    if not limits.continue_other_graphs_on_failure:
                        stop_new = True
    graph_reports: list[dict[str, Any]] = []
    for graph in plan.graphs:
        if graph.graph_id in failures:
            graph_root = root / "graphs" / graph.graph_id
            graph_root.mkdir(parents=True, exist_ok=True)
            failure = {"graph_id": graph.graph_id, "status": "failed",
                       "document_ids": list(graph.document_ids),
                       "failure_reason": failures[graph.graph_id][0],
                       "errors": failures[graph.graph_id]}
            write_json(graph_root / "terminal_failure.json", failure)
            graph_reports.append(failure)
            continue
        expected = sum(1 for item in items if item.graph_id == graph.graph_id)
        if len(results[graph.graph_id]) != expected:
            failure = {"graph_id": graph.graph_id, "status": "failed",
                       "document_ids": list(graph.document_ids),
                       "failure_reason": "graph extraction barrier was not satisfied",
                       "errors": ["graph extraction barrier was not satisfied"]}
            graph_root = root / "graphs" / graph.graph_id
            graph_root.mkdir(parents=True, exist_ok=True)
            write_json(graph_root / "terminal_failure.json", failure)
            graph_reports.append(failure)
            continue
        try:
            graph_reports.append(_finalize_graph(graph, results[graph.graph_id], plan=plan,
                settings=settings, bundle=bundle, root=root,
                embedding_resolver=embedding_resolver, neo4j_importer=neo4j_importer))
        except BaseException as error:
            failure = {"graph_id": graph.graph_id, "status": "failed",
                       "document_ids": list(graph.document_ids),
                       "failure_reason": f"{type(error).__name__}: {error}",
                       "errors": [f"{type(error).__name__}: {error}"]}
            write_json(root / "graphs" / graph.graph_id / "terminal_failure.json", failure)
            graph_reports.append(failure)
            if not limits.continue_other_graphs_on_failure:
                break
    completed = sum(row["status"] == "completed" for row in graph_reports)
    status = "completed_fully" if completed == len(plan.graphs) else (
        "completed_partially" if completed else "failed_entirely")
    accepted_results = [row for rows in results.values() for row in rows]
    attempts = sum(int(row["budget"]["primary_http_attempts"]) for row in accepted_results)
    prompt_tokens = sum(int(row["budget"]["prompt_tokens"]) for row in accepted_results)
    completion_tokens = sum(int(row["budget"]["completion_tokens"]) for row in accepted_results)
    document_reports = []
    for document in plan.documents:
        document_results = [row for row in results[document.graph_id]
                            if row["item"].document_id == document.document_id]
        document_failures = sum(item.document_id == document.document_id for item in failed_items)
        document_reports.append({
            "document_id": document.document_id, "graph_id": document.graph_id,
            "span_count": sum(item.document_id == document.document_id for item in items),
            "accepted_span_count": len(document_results),
            "failed_span_count": document_failures,
            "prompt_tokens": sum(int(row["budget"]["prompt_tokens"]) for row in document_results),
            "completion_tokens": sum(int(row["budget"]["completion_tokens"]) for row in document_results),
            "queue_wait_seconds": None, "processing_duration_seconds": None,
        })
        write_json(root / "documents" / document.document_id / "extraction_status.json", {
            "document_id": document.document_id, "graph_id": document.graph_id,
            "status": "passed" if len(document_results) == sum(
                item.document_id == document.document_id for item in items
            ) else "failed",
            "accepted_span_count": len(document_results),
            "failed_span_count": document_failures,
        })
    report = {"batch_id": plan.manifest.batch.id, "status": status,
              "documents_total": len(plan.documents), "graphs_total": len(plan.graphs),
              "graphs_completed": completed, "graphs_failed": len(plan.graphs) - completed,
              "spans_total": len(items), "peak_global_concurrency": scheduler.peak_in_flight,
              "model_attempts": attempts + len(failed_items),
              "accepted_extractions": len(accepted_results),
              "failed_extractions": len(failed_items),
              "input_tokens": prompt_tokens, "output_tokens": completion_tokens,
              "embedding_calls": sum(int(row.get("embedding_call_count") or 0)
                                     for row in graph_reports),
              "cached_input_tokens": None, "cache_hit_tokens": None,
              "cache_miss_tokens": None, "scheduled_span_requests": scheduled_count,
              "wall_clock_seconds": round(time.perf_counter() - started, 3),
              "graphs": graph_reports, "documents": document_reports}
    write_json(root / "batch_report.json", report)
    write_json(root / "batch_completion_status.json", {"status": status})
    return 0 if completed == len(plan.graphs) else 1
