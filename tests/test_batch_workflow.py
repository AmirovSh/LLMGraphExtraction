from __future__ import annotations

import json
from pathlib import Path

import yaml

from runtime.batch_runner import run_batch


ROOT = Path(__file__).resolve().parents[1]


class Response:
    status_code = 200
    content = b"{}"
    text = ""

    def json(self):
        arguments = {
            "status": "facts_present",
            "entities": [
                {"canonical_name": "Acme", "entity_type": "ORGANIZATION"},
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
    return resolution, {"embedder_calls": 1, "pair_decisions": []}


def fake_neo4j(_graph_id, _nodes, _edges, _settings, _ownership):
    return {"status": "projected"}, {"passed": True}


def test_hybrid_batch_completes_with_group_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTIC_GRAPH_LLM_BASE_URL", "http://offline")
    monkeypatch.setenv("SEMANTIC_GRAPH_EMBEDDING_BASE_URL", "http://offline")
    monkeypatch.setenv("OPENAI_API_KEY", "offline")
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

    def post(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        return Response()

    assert run_batch(path, ROOT / "config", llm_post=post,
                     embedding_resolver=fake_resolution,
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
    assert len(namespaces) == 3
