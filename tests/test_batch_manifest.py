from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from config.settings import load_project_settings
from runtime.batch_manifest import BatchManifest, resolve_batch_plan


ROOT = Path(__file__).resolve().parents[1]


def manifest(**updates):
    value = {
        "manifest_version": 1,
        "batch": {"id": "batch_1", "output_directory": "out"},
        "execution": {"global_max_in_flight": 4, "per_document_max_in_flight": 1,
                      "per_graph_max_in_flight": 2,
                      "scheduling_policy": "fair_round_robin",
                      "continue_other_graphs_on_failure": True},
        "model": {"provider_profile": "deepseek_v4_flash_0731_vllm",
                  "model_id": "deepseek-ai/DeepSeek-V4-Flash-0731"},
        "documents": [{"id": "d1", "path": "d1.txt"}, {"id": "d2", "path": "d2.txt"}],
        "graphs": [{"id": "g1", "documents": ["d1"]}, {"id": "g2", "documents": ["d2"]}],
    }
    value.update(updates)
    return value


@pytest.mark.parametrize("mutate", [
    lambda value: value["documents"].append({"id": "d1", "path": "x.txt"}),
    lambda value: value["graphs"].append({"id": "g1", "documents": ["d2"]}),
    lambda value: value["graphs"][0]["documents"].append("unknown"),
    lambda value: value["graphs"][1]["documents"].append("d1"),
    lambda value: value.update(graphs=[{"id": "g1", "documents": ["d1"]}]),
    lambda value: value["graphs"].append({"id": "empty", "documents": []}),
    lambda value: value.update(manifest_version=2),
])
def test_invalid_manifests_are_rejected(mutate):
    value = manifest()
    mutate(value)
    with pytest.raises(ValidationError):
        BatchManifest.model_validate(value)


def test_invalid_execution_limit_is_rejected():
    value = manifest()
    value["execution"]["per_graph_max_in_flight"] = 5
    with pytest.raises(ValidationError):
        BatchManifest.model_validate(value)


def test_resolution_rejects_missing_file_without_creating_output(tmp_path):
    path = tmp_path / "batch.yaml"
    path.write_text(yaml.safe_dump(manifest()), encoding="utf-8")
    settings = load_project_settings(ROOT / "config")
    with pytest.raises(FileNotFoundError):
        resolve_batch_plan(path, settings)
    assert not (tmp_path / "out").exists()


def test_hybrid_2_3_1_plan(tmp_path):
    documents = []
    for index in range(1, 7):
        name = f"d{index}"
        (tmp_path / f"{name}.txt").write_text(name, encoding="utf-8")
        documents.append({"id": name, "path": f"{name}.txt"})
    value = manifest(documents=documents, graphs=[
        {"id": "a", "documents": ["d1", "d2"]},
        {"id": "b", "documents": ["d3", "d4", "d5"]},
        {"id": "c", "documents": ["d6"]},
    ])
    path = tmp_path / "batch.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    plan = resolve_batch_plan(path, load_project_settings(ROOT / "config"))
    assert [len(graph.document_ids) for graph in plan.graphs] == [2, 3, 1]
    assert all(not str(document.path) in str(plan.public_dict()) for document in plan.documents)
