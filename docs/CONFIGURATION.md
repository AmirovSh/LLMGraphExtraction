# Configuration

All active non-secret values are loaded from `config/*.yaml`; secrets are read
only from the environment variables named there. Every run persists a redacted
`resolved_run_config.json`.

`llm_request_overrides` is not a generic payload escape hatch. Production allows
only the Kimi `chat_template_kwargs` keys `thinking` and
`preserve_thinking`; model, messages, tools, transport, endpoint, headers, and
credential-like keys are rejected before transport.

## Active extraction

```yaml
extraction:
  granularity: evidence_span
  contract_id: evidence_span_fact_extraction
  prompt_id: prompt_kimi_default
  max_concurrency: 4
  failure_policy: finish_in_flight_then_fail
```

The schema is `fact_extraction_schema`. It uses compact
zero-based entity/relation indices and top-level temporal bindings. Technical
IDs and evidence references are added deterministically after validation.

## Kimi provider

Production selects `kimi_k2_6_vllm_structured` and
`kimi_structured_temperature_zero`:

```yaml
temperature: 0.0
top_p: null
seed: null
```

Null `top_p` and `seed` are omitted from the request. The self-hosted Kimi
profile sends `chat_template_kwargs.thinking=false` and
`preserve_thinking=false`, uses `tool_choice=auto`, and requires the server's
native `kimi_k2` tool parser. The original chat template remains unchanged.

`kimi_structured_deterministic` is a backward-compatible alias only. Its name
does not promise determinism: temperature zero reduces sampling variation but
cannot guarantee identical semantic output on distributed serving backends.

The retained provider profiles are Kimi vLLM instant, structured production,
and thinking diagnostics. Qwen and DeepSeek qualification profiles are not
active configuration and cannot be selected.

## Entity resolution

- embedder: `bge-m3`;
- input: `canonical_name_only`;
- candidate threshold: `0.76`;
- automatic merge threshold: `0.90`;
- type: trace-only, not a hard filter;
- small runs: full pairwise cosine matrix;
- large runs: top-k selected from the full matrix (ANN is not implemented).

## Neo4j structural type names

`config/neo4j.yaml` defines the external projection syntax:

```yaml
graph_projection:
  entity_label: FACT_ENTITY
  relation_type: FACT_RELATION
```

Both values must match `^[A-Z][A-Z0-9_]{0,63}$`. They are validated without
normalization during settings loading, before model calls, artifacts, or Neo4j
access. Only validated identifiers are rendered into Cypher; all data values
remain parameters.

These settings do not change JSON fields, semantic `primary_type`,
`relation_family`, `raw_relation`, evidence, qualifiers, temporal data, node
IDs, or edge IDs. Each run records the selected names in
`projection_manifest.json`; rebuild rejects configuration that differs from the
recorded projection.

## Production completion versus benchmark

`completion_status.json` contains extraction, schema, evidence, FinalGraph,
projection, and parity status only. Development semantic metrics are written
separately by:

```powershell
python -m devtools.check_public_golden --live --run-id <run-id> --non-blocking
```

This creates `development_golden_report.json` and performs no model call.
