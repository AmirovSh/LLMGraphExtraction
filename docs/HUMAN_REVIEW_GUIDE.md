# Human review guide

This guide is for a technical review of an experimental release candidate. It
is not a production-readiness claim.

## What to review

Assess the architecture boundaries, configuration safety, one-call extraction
contract, document and graph-group isolation, observable entity resolution,
FinalGraph validation, artifact provenance, Neo4j parity, failure behavior,
DeepSeek/BGE-M3 compatibility, tests, and operational documentation.

## Five-minute repository map

- `config/`: strict settings models and public YAML examples. Start with
  `config/settings.py`, `config/runtime.yaml`, and `config/batch.example.yaml`.
- `prompts/`: versioned prompt registry and Pydantic extraction contracts.
- `runtime/`: deterministic orchestration and adapters. The main paths are
  `production_runner.py`, `batch_manifest.py`, `batch_scheduler.py`,
  `batch_runner.py`, `observable_entity_resolution.py`,
  `final_graph_contract.py`, and `json_neo4j_parity.py`.
- `scripts/`: CLI entry points: `run_production.py`, `run_batch.py`,
  `rebuild_projection.py`, `open_graph.py`, and
  `probe_model_compatibility.py`.
- `devtools/`: offline conformance and immutable public-golden release gates;
  these are not production post-processors.
- `tests/`: unit, contract, workflow, failure, parity, and public-golden tests.
- `docs/`: architecture, configuration, grouped-batch, graph-inspection, and
  acceptance documentation.

## Suggested review order

1. `README.md`
2. `docs/ARCHITECTURE.md`
3. `docs/BATCH_PROCESSING.md`
4. `config/settings.py`, `config/runtime.yaml`, and `config/batch.example.yaml`
5. `runtime/production_runner.py` and `scripts/run_production.py`
6. `runtime/batch_manifest.py`, `runtime/batch_scheduler.py`, and
   `runtime/batch_runner.py`
7. `runtime/production_inputs.py`, `runtime/transport_recovery.py`, and
   `prompts/contracts.py`
8. `runtime/observable_entity_resolution.py`
9. `runtime/final_graph_contract.py`
10. `runtime/json_neo4j_parity.py` and `runtime/projection_manifest.py`
11. The focused tests below
12. `docs/LIVE_ACCEPTANCE_SUMMARY.md`

## Offline verification commands

```powershell
python -m pytest
python -m devtools.check_project_conformance --offline
python -m devtools.check_public_golden --offline
python -m scripts.benchmark_batch_scheduler
docker compose -f docker-compose.neo4j.yml config
```

These commands require no LLM or embedding call. Docker Compose validation does
not start Neo4j.

## Focused reviewer commands

```powershell
python -m pytest tests/test_batch_manifest.py
python -m pytest tests/test_batch_manifest.py::test_hybrid_2_3_1_plan
python -m pytest tests/test_batch_scheduler.py
python -m pytest tests/test_batch_workflow.py::test_hybrid_batch_completes_with_group_isolation
python -m pytest tests/test_batch_workflow.py::test_one_document_batch_is_semantically_equivalent_to_legacy_run
python -m pytest tests/test_provider_compatibility.py::test_probe_payload_uses_explicit_served_model_alias tests/test_provider_compatibility.py::test_tool_probe_validates_nested_argument_types
python -m pytest tests/test_projection_schema_parity.py::test_projection_contract_reports_data_schema_and_namespace_parity
```

## Minimal single-document walkthrough

After configuring the documented LLM, embedding, and Neo4j environment names,
use the tracked neutral sample and a new output directory:

```powershell
python -m scripts.run_production `
  --input examples/sample_input.txt `
  --output outputs/review_single
```

`fact_graph.json` is the authoritative graph. The run also records source,
prompt, schema, request accounting, entity resolution, validation, projection,
parity, and completion artifacts. `projection_manifest.json` binds the Neo4j
labels/types; Neo4j is the derived projection. Raw request/response artifacts
may contain source text.

## Minimal grouped-batch walkthrough

`config/batch.example.yaml` demonstrates the 2 + 3 + 1 topology: two documents
in graph A, three in graph B, and one in graph C. Create the six safe relative
input files referenced by the example, then run:

```powershell
python -m scripts.run_batch --manifest config/batch.example.yaml
```

A document owns its evidence spans; a graph group owns embedding, entity
resolution, FinalGraph, and one Neo4j namespace; the batch owns fair global
scheduling and aggregate reporting. A failed group can be isolated while other
groups finish when the manifest enables continuation. Entities are never
resolved across graph groups. A one-document batch remains compatible with the
single-document semantic path.

## Artifact authority

- `fact_graph.json` is authoritative; Neo4j is derived.
- `projection_manifest.json` records validated Neo4j structural labels/types.
- `json_neo4j_edge_diff.json` and validation reports prove projection equality.
- Raw request/response artifacts may contain source-document content and must
  be handled as sensitive run data.

## Live acceptance evidence

The sanitized evidence is in `docs/LIVE_ACCEPTANCE_SUMMARY.md`. The explicitly
configured DeepSeek served alias passed native and complex tool calls; BGE-M3
returned 1024-dimensional vectors; the single-document smoke and live 2 + 3 + 1
batch passed; all three projections passed parity; and no retry or fallback
occurred. Private endpoints, credentials, inputs, and raw responses are not
tracked.

## Known limitations

- Semantic recall is not guaranteed; the reviewed fixture recalled 10 of 11
  expected facts and is not a universal quality score.
- Temperature zero does not guarantee byte-identical semantic output.
- Queue-wait and per-document processing timing are not instrumented.
- The deployed vLLM version and flags may not be observable.
- An evidence-backed alias/equivalence edge can become a redundant self-loop
  after entity consolidation. It remains structurally valid with provenance and
  is not removed using lexical raw-relation heuristics.
- Raw artifacts may contain source-document content.
- Resume is unsupported; use a new output directory.
- This project remains an experimental release candidate.
