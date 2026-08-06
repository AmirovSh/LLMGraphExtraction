from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest
import yaml

from runtime.batch_runner import run_batch
from runtime.production_runner import run as run_single


ROOT = Path(__file__).resolve().parents[1]


class Response:
    status_code = 200
    content = b"{}"
    text = ""

    def __init__(self, entity_name="Acme"):
        self.entity_name = entity_name

    def json(self):
        arguments = {
            "status": "facts_present",
            "entities": [
                {"canonical_name": self.entity_name, "entity_type": "ORGANIZATION"},
                {"canonical_name": "Office", "entity_type": "PLACE"},
            ],
            "relations": [{
                "source_entity_index": 0, "target_entity_index": 1,
                "relation_family": "RESPONSIBILITY_OWNERSHIP",
                "raw_relation": "operates", "condition": None, "modality": "ASSERTED",
                "quantity": None, "version": None, "negated": False,
            }],
            "temporal_bindings": [], "no_graph_fact_reason": None,
        }
        return {"choices": [{"finish_reason": "tool_calls", "message": {
            "tool_calls": [{"function": {
                "name": "submit_dynamic_inventory",
                "arguments": json.dumps(arguments),
            }}],
        }}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    def raise_for_status(self):
        return None


def fake_resolution(entities, _relations, **_kwargs):
    by_name = {}
    resolution = {}
    for entity in entities:
        by_name.setdefault(entity["name"], f"E{len(by_name) + 1:03d}")
        resolution[entity["local_id"]] = by_name[entity["name"]]
    return resolution, {"embedder_calls": 1, "pair_decisions": [],
        "merged_pairs": [], "keep_separate_pairs": [],
        "unresolved_merge_candidates": [], "candidate_threshold": .76,
        "merge_threshold": .90, "vector_dimension": 2,
        "unique_pair_count": max(0, len(entities) * (len(entities) - 1) // 2)}


def fake_neo4j(_graph_id, _nodes, _edges, _settings, _ownership):
    return {"status": "projected"}, {"status": "passed", "passed": True}


def test_hybrid_batch_completes_with_group_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTIC_GRAPH_LLM_BASE_URL", "http://offline")
    monkeypatch.setenv("SEMANTIC_GRAPH_EMBEDDING_BASE_URL", "http://offline")
    monkeypatch.setenv("OPENAI_API_KEY", "offline")
    monkeypatch.setenv("NEO4J_URI", "neo4j://offline")
    monkeypatch.setenv("NEO4J_USERNAME", "offline")
    monkeypatch.setenv("NEO4J_PASSWORD", "offline")
    documents = []
    for index in range(1, 7):
        path = tmp_path / f"d{index}.txt"
        path.write_text("Acme operates Office.", encoding="utf-8")
        documents.append({"id": f"d{index}", "path": path.name})
    output = tmp_path / "output"
    manifest = {
        "manifest_version": 1,
        "batch": {"id": "hybrid", "output_directory": str(output)},
        "execution": {"global_max_in_flight": 4, "per_document_max_in_flight": 1,
                      "per_graph_max_in_flight": 2,
                      "scheduling_policy": "fair_round_robin",
                      "continue_other_graphs_on_failure": True},
        "model": {"provider_profile": "deepseek_v4_flash_0731_vllm",
                  "model_id": "deepseek-ai/DeepSeek-V4-Flash-0731"},
        "documents": documents,
        "graphs": [
            {"id": "a", "documents": ["d1", "d2"]},
            {"id": "b", "documents": ["d3", "d4", "d5"]},
            {"id": "c", "documents": ["d6"]},
        ],
    }
    path = tmp_path / "batch.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    attempts = 0
    resolution_scopes = []

    def post(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        return Response()

    def scoped_resolution(entities, relations, **kwargs):
        resolution_scopes.append({span.split(":", 1)[0]
                                  for entity in entities
                                  for span in entity["evidence_span_ids"]})
        return fake_resolution(entities, relations, **kwargs)

    assert run_batch(path, ROOT / "config", llm_post=post,
                     embedding_resolver=scoped_resolution,
                     neo4j_importer=fake_neo4j) == 0
    assert attempts == 6
    report = json.loads((output / "batch_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed_fully"
    assert [row["span_count"] for row in report["graphs"]] == [2, 3, 1]
    assert report["peak_global_concurrency"] <= 4
    namespaces = set()
    for graph_id, document_ids in (("a", {"d1", "d2"}), ("b", {"d3", "d4", "d5"}), ("c", {"d6"})):
        root = output / "graphs" / graph_id
        graph = json.loads((root / "fact_graph.json").read_text(encoding="utf-8"))
        graph_manifest = json.loads((root / "graph_manifest.json").read_text(encoding="utf-8"))
        namespaces.add(graph_manifest["neo4j_namespace"])
        assert set(graph_manifest["document_ids"]) == document_ids
        assert all(span.split(":", 1)[0] in document_ids
                   for node in graph["nodes"] for span in node["evidence_span_ids"])
        assert len(graph["nodes"]) == 2
    assert len(namespaces) == 3
    assert resolution_scopes == [{"d1", "d2"}, {"d3", "d4", "d5"}, {"d6"}]


def test_one_document_batch_is_semantically_equivalent_to_legacy_run(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTIC_GRAPH_LLM_BASE_URL", "http://offline")
    monkeypatch.setenv("SEMANTIC_GRAPH_EMBEDDING_BASE_URL", "http://offline")
    monkeypatch.setenv("OPENAI_API_KEY", "offline")
    monkeypatch.setenv("NEO4J_URI", "neo4j://offline")
    monkeypatch.setenv("NEO4J_USERNAME", "offline")
    monkeypatch.setenv("NEO4J_PASSWORD", "offline")
    source = tmp_path / "source.txt"
    source.write_text("Acme operates Office.", encoding="utf-8")
    single_output = tmp_path / "single"
    assert run_single(source, single_output, ROOT / "config", run_id="compat",
                      llm_post=lambda *_args, **_kwargs: Response(),
                      embedding_resolver=fake_resolution,
                      neo4j_importer=fake_neo4j) == 0
    batch_output = tmp_path / "batch_output"
    manifest = {
        "manifest_version": 1,
        "batch": {"id": "compat", "output_directory": str(batch_output)},
        "execution": {"global_max_in_flight": 1, "per_document_max_in_flight": 1,
                      "per_graph_max_in_flight": 1,
                      "scheduling_policy": "fair_round_robin",
                      "continue_other_graphs_on_failure": True},
        "model": {"provider_profile": "deepseek_v4_flash_0731_vllm",
                  "model_id": "deepseek-ai/DeepSeek-V4-Flash-0731"},
        "documents": [{"id": "document", "path": source.name}],
        "graphs": [{"id": "graph", "documents": ["document"]}],
    }
    path = tmp_path / "compat.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    assert run_batch(path, ROOT / "config",
                     llm_post=lambda *_args, **_kwargs: Response(),
                     embedding_resolver=fake_resolution,
                     neo4j_importer=fake_neo4j) == 0
    legacy = json.loads((single_output / "fact_graph.json").read_text(encoding="utf-8"))
    grouped = json.loads((batch_output / "graphs" / "graph" / "fact_graph.json").read_text(encoding="utf-8"))
    def normalized(graph):
        evidence = lambda values: tuple(sorted(value.split(":")[-1] for value in values))
        names = {node["entity_id"]: (node["canonical_name"], node["primary_type"])
                 for node in graph["nodes"]}
        nodes = sorted((node["canonical_name"], node["primary_type"],
                        tuple(node["aliases"]), evidence(node["evidence_span_ids"]))
                       for node in graph["nodes"])
        edges = sorted((names[edge["source_entity_id"]], names[edge["target_entity_id"]],
                        edge["raw_relation"], edge["relation_family"],
                        edge["relation_description"], edge["support_level"],
                        json.dumps(edge["qualifiers"], sort_keys=True),
                        evidence(edge["evidence_span_ids"]))
                       for edge in graph["edges"])
        return nodes, edges

    assert normalized(legacy) == normalized(grouped)


@pytest.mark.parametrize("combined", [True, False])
def test_two_documents_produce_one_or_two_graphs(tmp_path, monkeypatch, combined):
    monkeypatch.setenv("SEMANTIC_GRAPH_LLM_BASE_URL", "http://offline")
    monkeypatch.setenv("SEMANTIC_GRAPH_EMBEDDING_BASE_URL", "http://offline")
    monkeypatch.setenv("OPENAI_API_KEY", "offline")
    for name in ("d1", "d2"):
        (tmp_path / f"{name}.txt").write_text("Acme operates Office.", encoding="utf-8")
    output = tmp_path / "out"
    graphs = ([{"id": "combined", "documents": ["d1", "d2"]}] if combined else
              [{"id": "one", "documents": ["d1"]},
               {"id": "two", "documents": ["d2"]}])
    payload = {
        "manifest_version": 1, "batch": {"id": "pair", "output_directory": str(output)},
        "execution": {"global_max_in_flight": 2, "per_document_max_in_flight": 1,
                      "per_graph_max_in_flight": 2, "scheduling_policy": "fair_round_robin",
                      "continue_other_graphs_on_failure": True},
        "model": {"provider_profile": "deepseek_v4_flash_0731_vllm",
                  "model_id": "deepseek-ai/DeepSeek-V4-Flash-0731"},
        "documents": [{"id": name, "path": f"{name}.txt"} for name in ("d1", "d2")],
        "graphs": graphs,
    }
    path = tmp_path / "pair.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    assert run_batch(path, ROOT / "config", llm_post=lambda *_a, **_k: Response(),
                     embedding_resolver=fake_resolution, neo4j_importer=fake_neo4j) == 0
    report = json.loads((output / "batch_report.json").read_text(encoding="utf-8"))
    assert report["graphs_completed"] == (1 if combined else 2)
    assert report["model_attempts"] == 2


def test_completion_order_does_not_route_results_to_wrong_document(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTIC_GRAPH_LLM_BASE_URL", "http://offline")
    monkeypatch.setenv("SEMANTIC_GRAPH_EMBEDDING_BASE_URL", "http://offline")
    monkeypatch.setenv("OPENAI_API_KEY", "offline")
    (tmp_path / "slow.txt").write_text("SLOW operates Office.", encoding="utf-8")
    (tmp_path / "fast.txt").write_text("FAST operates Office.", encoding="utf-8")
    output = tmp_path / "ordered"
    payload = {
        "manifest_version": 1, "batch": {"id": "ordered", "output_directory": str(output)},
        "execution": {"global_max_in_flight": 2, "per_document_max_in_flight": 1,
                      "per_graph_max_in_flight": 2, "scheduling_policy": "fair_round_robin",
                      "continue_other_graphs_on_failure": True},
        "model": {"provider_profile": "deepseek_v4_flash_0731_vllm",
                  "model_id": "deepseek-ai/DeepSeek-V4-Flash-0731"},
        "documents": [{"id": "slow", "path": "slow.txt"}, {"id": "fast", "path": "fast.txt"}],
        "graphs": [{"id": "graph", "documents": ["slow", "fast"]}],
    }
    path = tmp_path / "ordered.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    def post(*_args, **kwargs):
        prompt = kwargs["json"]["messages"][-1]["content"]
        if "SLOW" in prompt:
            time.sleep(.05)
            return Response("SlowEntity")
        return Response("FastEntity")

    assert run_batch(path, ROOT / "config", llm_post=post,
                     embedding_resolver=fake_resolution, neo4j_importer=fake_neo4j) == 0
    graph = json.loads((output / "graphs" / "graph" / "fact_graph.json").read_text(encoding="utf-8"))
    provenance = {node["canonical_name"]: set(node["evidence_span_ids"]) for node in graph["nodes"]}
    assert provenance["SlowEntity"] == {"slow:S001"}
    assert provenance["FastEntity"] == {"fast:S001"}


def test_failed_group_skips_embedding_and_neo4j_while_other_group_completes(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTIC_GRAPH_LLM_BASE_URL", "http://offline")
    monkeypatch.setenv("SEMANTIC_GRAPH_EMBEDDING_BASE_URL", "http://offline")
    monkeypatch.setenv("OPENAI_API_KEY", "offline")
    (tmp_path / "bad.txt").write_text("FAIL.", encoding="utf-8")
    (tmp_path / "good.txt").write_text("GOOD.", encoding="utf-8")
    output = tmp_path / "failure"
    payload = {
        "manifest_version": 1, "batch": {"id": "failure", "output_directory": str(output)},
        "execution": {"global_max_in_flight": 2, "per_document_max_in_flight": 1,
                      "per_graph_max_in_flight": 1, "scheduling_policy": "fair_round_robin",
                      "continue_other_graphs_on_failure": True},
        "model": {"provider_profile": "deepseek_v4_flash_0731_vllm",
                  "model_id": "deepseek-ai/DeepSeek-V4-Flash-0731"},
        "documents": [{"id": "bad", "path": "bad.txt"}, {"id": "good", "path": "good.txt"}],
        "graphs": [{"id": "bad_graph", "documents": ["bad"]},
                   {"id": "good_graph", "documents": ["good"]}],
    }
    path = tmp_path / "failure.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    embedded = []
    projected = []

    def post(*_args, **kwargs):
        if "FAIL" in kwargs["json"]["messages"][-1]["content"]:
            raise httpx.ConnectError("offline failure")
        return Response()

    def embedding(entities, relations, **kwargs):
        embedded.append({span.split(":", 1)[0] for entity in entities
                         for span in entity["evidence_span_ids"]})
        return fake_resolution(entities, relations, **kwargs)

    def neo4j(graph_id, *args):
        projected.append(graph_id)
        return fake_neo4j(graph_id, *args)

    assert run_batch(path, ROOT / "config", llm_post=post,
                     embedding_resolver=embedding, neo4j_importer=neo4j) == 1
    assert embedded == [{"good"}]
    assert len(projected) == 1 and "good_graph" in projected[0]
    assert not (output / "graphs" / "bad_graph" / "fact_graph.json").exists()
    assert (output / "graphs" / "bad_graph" / "terminal_failure.json").is_file()
    assert (output / "graphs" / "good_graph" / "completion_status.json").is_file()
