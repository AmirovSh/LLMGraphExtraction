from __future__ import annotations

import json
import shutil
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import httpx

from config.settings import load_project_settings
from runtime.artifact_store import write_json
from runtime.json_neo4j_parity import build_parity_diff
from runtime.production_runner import run

ROOT = Path(__file__).resolve().parents[1]


class FakeResponse:
    status_code = 200
    content = b"response"
    text = ""

    def __init__(self, body: dict): self._body = body
    def json(self) -> dict: return self._body
    def raise_for_status(self) -> None: return None


def extraction_body() -> dict:
    arguments = {
        "status": "facts_present",
        "no_graph_fact_reason": None,
        "entities": [
            {"canonical_name": "Release Platform", "entity_type": "COMPONENT"},
            {"canonical_name": "Audit Console", "entity_type": "COMPONENT"},
        ],
        "relations": [{
            "source_entity_index": 0,
            "raw_relation": "exposes",
            "relation_family": "DATA_FLOW",
            "target_entity_index": 1,
            "condition": None,
            "modality": "ASSERTED",
            "quantity": None,
            "version": None,
            "negated": False,
        }],
        "temporal_bindings": [],
    }
    return {"choices":[{"finish_reason":"tool_calls","message":{"tool_calls":[{"function":{"name":"submit_dynamic_inventory","arguments":json.dumps(arguments)}}]}}],
            "usage":{"prompt_tokens":100,"completion_tokens":40,"total_tokens":140}}


def fake_resolution(entities, relations, **kwargs):
    kwargs["write_artifact"]("entity_embedding_inputs.json", {"inputs":[],"input_count":2})
    mapping = {"M001":"E001","M002":"E002"}
    unresolved={"left_local_id":"M001","right_local_id":"M002","similarity":.8,"decision":"UNRESOLVED","reason":"golden characterization"}
    report = {"embedder_calls":1,"usage":{"prompt_tokens":2},"retrieval_method":"full_pairwise_cosine_matrix","top_k":None,
              "candidate_threshold":.76,"merge_threshold":.90,"candidate_pairs":[unresolved],"merged_pairs":[],"keep_separate_pairs":[],
              "unresolved_merge_candidates":[unresolved],"all_pair_decision_counts":{"UNRESOLVED":1},"vector_dimension":1024,"embedding_input_count":2,"unique_pair_count":1}
    return mapping, report


def fake_neo4j(graph_id, nodes, edges, settings, ownership):
    edge_map = {edge["edge_id"]:(edge["source_entity_id"],edge["target_entity_id"]) for edge in edges}
    diff = build_parity_diff(graph_id=graph_id,json_nodes=nodes,json_edges=edges,neo4j_node_ids={node["entity_id"] for node in nodes},
                             neo4j_edges=edge_map,edges_without_graph_id=[],duplicate_edges=[],import_errors=[],ownership=ownership)
    return {"graph_id":graph_id,"nodes":len(nodes),"edges":len(edges)}, diff


def make_offline_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    sandbox = tmp_path / "repo"; shutil.copytree(ROOT / "config", sandbox / "config")
    source = sandbox / "source.txt"; source.write_text("The Release Platform exposes the Audit Console.\n",encoding="utf-8")
    output = sandbox / "run_golden"; monkeypatch.setattr("runtime.production_runner.ROOT",sandbox)
    monkeypatch.setenv("SEMANTIC_GRAPH_LLM_BASE_URL","https://offline.invalid"); monkeypatch.setenv("OPENAI_API_KEY","offline")
    monkeypatch.setenv("SEMANTIC_GRAPH_EMBEDDING_BASE_URL","https://offline.invalid")
    monkeypatch.setenv("NEO4J_URI","bolt://offline.invalid:7687")
    run(source,output,sandbox/"config",run_id="golden",llm_post=lambda *a,**k:FakeResponse(extraction_body()),embedding_resolver=fake_resolution,neo4j_importer=fake_neo4j)
    return source, output


def test_cli_wrapper_delegates_to_production_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.run_production as cli
    captured = {}
    monkeypatch.setattr(cli,"run",lambda *args,**kwargs: captured.update(args=args,kwargs=kwargs) or 17)
    monkeypatch.setattr(sys,"argv",["run_production","--input",str(tmp_path/"in.txt"),"--output",str(tmp_path/"out")])
    assert cli.main() == 17
    assert captured["args"][:2] == (tmp_path/"in.txt",tmp_path/"out")


def test_cli_reports_final_technical_production_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import scripts.run_production as cli
    output = tmp_path / "completed"
    output.mkdir()
    (output / "fact_graph.json").write_text(
        json.dumps({"source": {"units": [{"unit_id": "U001"}]}}),
        encoding="utf-8",
    )
    (output / "graph_view.json").write_text(
        json.dumps({"browser_url": "http://localhost/browser"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "run", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        cli, "load_successful_run",
        lambda _path: SimpleNamespace(
            run_id="completed", graph_id="fact_extraction_completed",
            node_count=2, relationship_count=1,
        ),
    )
    monkeypatch.setattr(
        sys, "argv",
        ["run_production", "--input", str(tmp_path/"input.txt"), "--output", str(output)],
    )
    assert cli.main() == 0
    stdout = capsys.readouterr().out
    for expected in (
        "PRODUCTION GRAPH BUILD COMPLETE", "Run ID: completed", "Units: 1",
        "Nodes: 2", "Edges: 1", "Neo4j projection: passed", "Parity: passed",
    ):
        assert expected in stdout


def test_production_entrypoint_has_no_development_golden_dependency() -> None:
    for path in (ROOT/"scripts"/"run_production.py", ROOT/"runtime"/"production_runner.py"):
        source = path.read_text(encoding="utf-8")
        assert "public_" + "golden" not in source
        assert "required_entities" not in source
        assert "required_relations" not in source
        assert "forbidden_relations" not in source


def test_atomic_json_write_preserves_previous_file_if_replace_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target=tmp_path/"critical.json"; target.write_text('{"old":true}\n',encoding="utf-8")
    monkeypatch.setattr("runtime.artifact_store.os.replace",lambda *_: (_ for _ in ()).throw(OSError("replace failed")))
    with pytest.raises(OSError,match="replace failed"): write_json(target,{"new":True})
    assert target.read_text(encoding="utf-8") == '{"old":true}\n'
    assert not list(tmp_path.glob("*.tmp"))


def test_offline_span_run_preserves_graph_resolution_and_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source,output=make_offline_run(tmp_path,monkeypatch)
    names=["fact_graph.json","prompt_manifest.json","projection_manifest.json","entity_resolution.json","global_entities.json","validation_report.json","run_report.json","completion_status.json","resolved_run_config.json","json_neo4j_edge_diff.json","model_call_budget.json","model_call_attempts.json"]
    before={name:json.loads((output/name).read_text(encoding="utf-8")) for name in names}
    graph=before["fact_graph.json"]; assert [node["entity_id"] for node in graph["nodes"]] == ["E001","E002"]
    assert graph["graph_id"].startswith("fact_extraction_golden_")
    assert graph["run_id"] == "golden"
    assert len(graph["manifest_identity_hash"]) == 64
    assert graph["source"]["source_name"] == "source.txt"
    assert graph["source"]["source_location_policy"] == "repository_relative"
    assert graph["source"]["repository_relative_path"] == "source.txt"
    assert len(graph["source"]["source_sha256"]) == 64
    assert graph["prompt_id"] == "prompt_kimi_default"
    assert "prompt_family" not in graph and "prompt_version" not in graph
    assert before["prompt_manifest.json"]["prompt_id"] == "prompt_kimi_default"
    assert len(before["prompt_manifest.json"]["prompt_hash"]) == 64
    assert set(before["prompt_manifest.json"]) == {
        "prompt_id", "prompt_hash",
        "contract_id", "schema_id", "request_identity",
    }
    assert before["prompt_manifest.json"]["contract_id"] == "evidence_span_fact_extraction"
    assert before["prompt_manifest.json"]["schema_id"] == "fact_extraction_schema"
    request_identity = before["prompt_manifest.json"]["request_identity"]
    assert request_identity["model_id"] == "kimi-2.6"
    assert request_identity["provider_profile_id"] == "kimi_k2_6_vllm_structured"
    assert request_identity["prompt_hash"] == before["prompt_manifest.json"]["prompt_hash"]
    assert request_identity["schema_hash"] == "eb4ce676614fd8e107c44563a487e0c16ba9d95ee047109bfc0d05f8c00ed3ae"
    assert request_identity["tool_name"] == "submit_dynamic_inventory"
    assert len(request_identity["allowed_extra_body_hash"]) == 64
    assert len(graph["edges"]) == 1 and graph["edges"][0]["source_entity_id"] == "E001" and graph["edges"][0]["target_entity_id"] == "E002"
    assert graph["edges"][0]["relation_family"] == "DATA_FLOW" and graph["edges"][0]["qualifiers"]["modality"] == "ASSERTED"
    assert before["entity_resolution.json"]["local_to_global"] == {"M001":"E001","M002":"E002"}
    assert before["entity_resolution.json"]["candidate_pairs"][0]["decision"] == "UNRESOLVED"
    assert before["model_call_budget.json"]["totals"] == {"primary_successful_calls":1,"primary_http_attempts":1,"primary_failed_attempts":0,"primary_timeout_attempts":0,"transport_successes":1,"transport_failures":0,"provider_response_rejections":0,"tool_contract_failures":0,"schema_failures":0,"evidence_failures":0,"accepted_extractions":1,"automatic_retries":0,"post_extraction_llm_calls":0,"embedding_calls":1,"prompt_tokens":100,"completion_tokens":40,"total_tokens":140,"thinking_responses":0,"schema_valid_extractions":1,"evidence_valid_extractions":1,"persisted_span_units":1}
    assert before["completion_status.json"] == {
        "run_status": "completed",
        "extraction_status": "passed",
        "schema_status": "passed",
        "evidence_status": "passed",
        "final_graph_status": "passed",
        "projection_status": "passed",
        "parity_status": "passed",
    }
def test_existing_output_fails_before_external_calls_or_artifact_writes(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("Source text.", encoding="utf-8")
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "owned.txt"
    marker.write_text("unchanged", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Resume is not supported"):
        run(
            source, output, ROOT / "config",
            llm_post=lambda *_a, **_k: pytest.fail("existing output made LLM call"),
            embedding_resolver=lambda *_a, **_k: pytest.fail("existing output made embedding call"),
            neo4j_importer=lambda *_a, **_k: pytest.fail("existing output mutated Neo4j"),
        )
    assert list(output.iterdir()) == [marker]


def test_external_source_succeeds_without_persisting_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "external-contract.txt"
    source.write_text(
        "The Release Platform exposes the Audit Console.", encoding="utf-8",
    )
    output = tmp_path / "external-output"
    monkeypatch.setenv("SEMANTIC_GRAPH_LLM_BASE_URL", "https://offline.invalid")
    monkeypatch.setenv("OPENAI_API_KEY", "offline")
    monkeypatch.setenv("SEMANTIC_GRAPH_EMBEDDING_BASE_URL", "https://offline.invalid")
    monkeypatch.setenv("NEO4J_URI", "bolt://offline.invalid:7687")
    run(
        source, output, ROOT / "config", run_id="external-source",
        llm_post=lambda *_a, **_k: FakeResponse(extraction_body()),
        embedding_resolver=fake_resolution, neo4j_importer=fake_neo4j,
    )
    graph_text = (output / "fact_graph.json").read_text(encoding="utf-8")
    graph = json.loads(graph_text)
    assert graph["source"]["source_location_policy"] == "external"
    assert graph["source"]["source_name"] == source.name
    assert graph["source"]["repository_relative_path"] is None
    assert str(tmp_path.resolve()) not in graph_text


def test_same_output_basename_generates_distinct_graph_namespaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.txt"
    source.write_text(
        "The Release Platform exposes the Audit Console.", encoding="utf-8",
    )
    monkeypatch.setenv("SEMANTIC_GRAPH_LLM_BASE_URL", "https://offline.invalid")
    monkeypatch.setenv("OPENAI_API_KEY", "offline")
    monkeypatch.setenv("SEMANTIC_GRAPH_EMBEDDING_BASE_URL", "https://offline.invalid")
    monkeypatch.setenv("NEO4J_URI", "bolt://offline.invalid:7687")
    graph_ids = []
    for parent in (tmp_path / "first", tmp_path / "second"):
        output = parent / "same-name"
        run(
            source, output, ROOT / "config",
            llm_post=lambda *_a, **_k: FakeResponse(extraction_body()),
            embedding_resolver=fake_resolution, neo4j_importer=fake_neo4j,
        )
        graph_ids.append(json.loads(
            (output / "fact_graph.json").read_text(encoding="utf-8")
        )["graph_id"])
    assert graph_ids[0] != graph_ids[1]


def test_unreadable_or_empty_input_fails_before_output_creation(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    missing_output = tmp_path / "missing-output"
    with pytest.raises(FileNotFoundError):
        run(missing, missing_output, ROOT / "config", run_id="missing")
    assert not missing_output.exists()

    empty = tmp_path / "empty.txt"
    empty.write_text("  \n", encoding="utf-8")
    empty_output = tmp_path / "empty-output"
    with pytest.raises(ValueError, match="no extractable units"):
        run(empty, empty_output, ROOT / "config", run_id="empty")
    assert not empty_output.exists()


def test_projection_failure_keeps_authoritative_json_without_completion_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.txt"
    source.write_text(
        "The Release Platform exposes the Audit Console.", encoding="utf-8",
    )
    output = tmp_path / "projection-failure"
    monkeypatch.setenv("SEMANTIC_GRAPH_LLM_BASE_URL", "https://offline.invalid")
    monkeypatch.setenv("OPENAI_API_KEY", "offline")
    monkeypatch.setenv("SEMANTIC_GRAPH_EMBEDDING_BASE_URL", "https://offline.invalid")
    monkeypatch.setenv("NEO4J_URI", "bolt://offline.invalid:7687")

    def fail_projection(*_args, **_kwargs):
        raise RuntimeError("injected projection failure")

    with pytest.raises(RuntimeError, match="injected projection failure"):
        run(
            source, output, ROOT / "config", run_id="projection-failure",
            llm_post=lambda *_a, **_k: FakeResponse(extraction_body()),
            embedding_resolver=fake_resolution, neo4j_importer=fail_projection,
        )
    assert (output / "fact_graph.json").is_file()
    assert not (output / "completion_status.json").exists()


def test_span_timeout_is_accounted_without_retry_or_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox=tmp_path/"repo"; shutil.copytree(ROOT/"config",sandbox/"config")
    source=sandbox/"source.txt"; source.write_text("The Release Platform exposes the Audit Console.\n",encoding="utf-8")
    output=sandbox/"run_timeout"; monkeypatch.setattr("runtime.production_runner.ROOT",sandbox)
    monkeypatch.setenv("SEMANTIC_GRAPH_LLM_BASE_URL","https://offline.invalid"); monkeypatch.setenv("OPENAI_API_KEY","offline")
    monkeypatch.setenv("SEMANTIC_GRAPH_EMBEDDING_BASE_URL","https://offline.invalid")
    monkeypatch.setenv("NEO4J_URI","bolt://offline.invalid:7687")

    with pytest.raises(RuntimeError, match="span-flat extraction failed"):
        run(source,output,sandbox/"config",llm_post=lambda *_a,**_k:(_ for _ in ()).throw(httpx.ReadTimeout("timed out")),
            embedding_resolver=fake_resolution,neo4j_importer=fake_neo4j)
    failed=json.loads((output/"model_call_attempts.json").read_text(encoding="utf-8"))["totals"]
    assert failed["primary_http_attempts"] == 1 and failed["primary_failed_attempts"] == 1 and failed["primary_timeout_attempts"] == 1

    assert failed["automatic_retries"] == 0
    assert failed["transport_failures"] == 1
    assert failed["transport_successes"] == 0


@pytest.mark.parametrize(
    ("case", "outcome", "subcategory"),
    [
        ("missing_choices", "provider_response_rejected", "missing_choices"),
        ("finish_reason", "provider_response_rejected", "invalid_finish_reason"),
        ("reasoning", "provider_response_rejected", "reasoning_policy_violation"),
        ("wrong_tool", "tool_contract_failed", "wrong_tool_name"),
        ("malformed_arguments", "tool_contract_failed", "malformed_arguments"),
        ("schema", "schema_failed", "pydantic_validation_failed"),
        ("evidence", "evidence_failed", "temporal_evidence_failed"),
    ],
)
def test_http_200_rejections_have_one_truthful_terminal_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    case: str, outcome: str, subcategory: str,
) -> None:
    source = tmp_path / f"{case}.txt"
    source.write_text(
        "The Release Platform exposes the Audit Console.", encoding="utf-8",
    )
    output = tmp_path / f"run-{case}"
    monkeypatch.setenv("SEMANTIC_GRAPH_LLM_BASE_URL", "https://offline.invalid")
    monkeypatch.setenv("OPENAI_API_KEY", "offline")
    body = extraction_body()
    if case == "missing_choices":
        body.pop("choices")
    elif case == "finish_reason":
        body["choices"][0]["finish_reason"] = "length"
    elif case == "reasoning":
        body["choices"][0]["message"]["reasoning_content"] = "not allowed"
    elif case == "wrong_tool":
        body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = "wrong"
    elif case == "malformed_arguments":
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{"
    else:
        arguments = json.loads(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )
        if case == "schema":
            arguments.pop("status")
        elif case == "evidence":
            arguments["temporal_bindings"] = [
                {"relation_index": 0, "surface": "not present in evidence"},
            ]
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps(arguments)

    with pytest.raises(RuntimeError, match="span-flat extraction failed"):
        run(
            source, output, ROOT / "config", run_id=f"rejected-{case}",
            llm_post=lambda *_a, **_k: FakeResponse(body),
            embedding_resolver=lambda *_a, **_k: pytest.fail("rejected output reached embeddings"),
            neo4j_importer=lambda *_a, **_k: pytest.fail("rejected output reached Neo4j"),
        )
    attempts = json.loads(
        (output / "model_call_attempts.json").read_text(encoding="utf-8")
    )
    unit = attempts["units"][0]
    totals = attempts["totals"]
    assert unit["terminal_outcome"] == outcome
    assert unit["terminal_subcategory"] == subcategory
    assert totals["primary_http_attempts"] == 1
    assert totals["transport_successes"] == 1
    assert totals["transport_failures"] == 0
    assert totals["accepted_extractions"] == 0
    assert totals[{"provider_response_rejected": "provider_response_rejections", "tool_contract_failed": "tool_contract_failures", "schema_failed": "schema_failures", "evidence_failed": "evidence_failures"}[outcome]] == 1


def test_production_help_does_not_expose_resume(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.run_production as cli
    monkeypatch.setattr(sys, "argv", ["run_production", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--resume" not in help_text


def test_resolved_config_contains_names_not_secret_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY","highly-secret-value")
    saved=load_project_settings(ROOT/"config").resolved_for_artifact(); serialized=json.dumps(saved)
    assert "highly-secret-value" not in serialized and "OPENAI_API_KEY" in serialized


def test_thinking_reasoning_is_not_persisted_in_production_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = tmp_path / "repo"
    shutil.copytree(ROOT / "config", sandbox / "config")
    source = sandbox / "source.txt"
    source.write_text(
        "The Release Platform exposes the Audit Console.\n", encoding="utf-8",
    )
    output = sandbox / "run_thinking"
    monkeypatch.setattr("runtime.production_runner.ROOT", sandbox)
    monkeypatch.setenv("SEMANTIC_GRAPH_LLM_BASE_URL", "https://offline.invalid")
    monkeypatch.setenv("OPENAI_API_KEY", "offline")
    monkeypatch.setenv(
        "SEMANTIC_GRAPH_EMBEDDING_BASE_URL", "https://offline.invalid",
    )
    monkeypatch.setenv("NEO4J_URI", "bolt://offline.invalid:7687")
    monkeypatch.setenv(
        "SEMANTIC_GRAPH_LLM_PROVIDER_PROFILE",
        "kimi_k2_6_vllm_thinking_structured",
    )
    body = extraction_body()
    body["choices"][0]["message"]["reasoning_content"] = (
        "private chain of thought"
    )
    body["usage"]["completion_tokens_details"] = {"reasoning_tokens": 17}
    run(
        source, output, sandbox / "config",
        llm_post=lambda *_a, **_k: FakeResponse(body),
        embedding_resolver=fake_resolution,
        neo4j_importer=fake_neo4j,
    )
    response = json.loads(
        (output / "raw_responses" / "U001_S001_response.json").read_text(
            encoding="utf-8",
        )
    )
    assert response["reasoning_present"] is True
    assert response["reasoning_tokens"] == 17
    assert "reasoning_content" not in response["body"]["choices"][0]["message"]
    assert "private chain of thought" not in json.dumps(response)
    budget = json.loads(
        (output / "model_call_budget.json").read_text(encoding="utf-8")
    )
    assert budget["totals"]["thinking_responses"] == 1


def test_invalid_final_graph_is_not_persisted_or_projected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox=tmp_path/"repo"; shutil.copytree(ROOT/"config",sandbox/"config")
    source=sandbox/"source.txt"; source.write_text(
        "The Release Platform exposes the Audit Console.\n", encoding="utf-8",
    )
    output=sandbox/"run_invalid"; monkeypatch.setattr("runtime.production_runner.ROOT",sandbox)
    monkeypatch.setenv("SEMANTIC_GRAPH_LLM_BASE_URL","https://offline.invalid")
    monkeypatch.setenv("OPENAI_API_KEY","offline")
    monkeypatch.setenv("SEMANTIC_GRAPH_EMBEDDING_BASE_URL","https://offline.invalid")
    body=extraction_body()
    arguments=json.loads(body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    arguments["relations"][0]["raw_relation"]=" "
    body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]=json.dumps(arguments)
    with pytest.raises(RuntimeError,match="before persistence"):
        run(
            source,output,sandbox/"config",
            llm_post=lambda *_a,**_k:FakeResponse(body),
            embedding_resolver=fake_resolution,
            neo4j_importer=lambda *_a,**_k:pytest.fail("invalid graph reached Neo4j"),
        )
    assert not (output/"fact_graph.json").exists()
    failure=json.loads((output/"terminal_failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "terminal_validation_failure"
    assert failure["rule_category"] == "final_graph_contract"
    assert failure["errors"][0]["loc"][-1] == "raw_relation"


def test_unknown_extraction_evidence_fails_before_embeddings_and_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox=tmp_path/"repo"; shutil.copytree(ROOT/"config",sandbox/"config")
    source=sandbox/"source.txt"; source.write_text(
        "The Release Platform exposes the Audit Console.\n", encoding="utf-8",
    )
    output=sandbox/"run_invalid_evidence"
    monkeypatch.setattr("runtime.production_runner.ROOT",sandbox)
    monkeypatch.setenv("SEMANTIC_GRAPH_LLM_BASE_URL","https://offline.invalid")
    monkeypatch.setenv("OPENAI_API_KEY","offline")
    monkeypatch.setenv("SEMANTIC_GRAPH_EMBEDDING_BASE_URL","https://offline.invalid")
    body=extraction_body()
    arguments=json.loads(body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    arguments["temporal_bindings"]=[{"relation_index":0,"surface":"not in evidence"}]
    body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]=json.dumps(arguments)
    with pytest.raises(RuntimeError,match="span-flat extraction failed"):
        run(
            source,output,sandbox/"config",
            llm_post=lambda *_a,**_k:FakeResponse(body),
            embedding_resolver=lambda *_a,**_k:pytest.fail("invalid evidence reached embeddings"),
            neo4j_importer=lambda *_a,**_k:pytest.fail("invalid evidence reached Neo4j"),
        )
    assert not (output/"fact_graph.json").exists()
    assert not (output/"completion_status.json").exists()
    failure=json.loads((output/"terminal_failure.json").read_text(encoding="utf-8"))
    assert failure["rule_category"] == "span_flat_extraction"
    assert "TEMPORAL_SURFACE_NOT_EVIDENCE_BACKED" in failure["errors"][0]
