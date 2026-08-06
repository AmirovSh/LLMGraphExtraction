# Production architecture

The supported pipeline is:

`text -> deterministic evidence spans -> one Kimi extraction per span -> structural validation -> bge-m3 -> observable entity resolution -> deterministic FinalGraph JSON -> Neo4j projection`

JSON is authoritative. Neo4j is an exact, graph-ID-scoped projection and is
never a recovery or semantic-editing store.

## Selected production contract

- model: Kimi K2.6 (`kimi-2.6`);
- provider profile: `kimi_k2_6_vllm_structured`;
- sampling profile: `kimi_structured_temperature_zero`;
- extraction contract: `evidence_span_fact_extraction`;
- prompt: `prompt_kimi_default`;
- schema: `fact_extraction_schema`;
- embedding model/input: `bge-m3` / `canonical_name_only`.

> [!NOTE]
> The currently selected `kimi-2.6` extraction model, `bge-m3` embedding model,
> prompt, thresholds, and other runtime parameters were used to provide a
> reproducible demonstration. They have not been established as optimal for
> this text or for other document classes. Do not treat the checked-in
> configuration as recommended quality, cost, or performance parameters;
> evaluate models, prompts, and thresholds against representative evidence and
> acceptance criteria for the intended deployment.

Temperature zero reduces sampling variation but does not guarantee identical
semantic output on a distributed serving backend. Byte-identical graphs,
identical node/edge counts, and repeated N/N semantic assertions are not
production or release gates.

## One-shot boundary

Each deterministic evidence span receives one primary request and one accepted
schema-valid result. The pipeline does not retry semantic extraction, vote over
outputs, select a more complete answer, merge multiple answers, run an LLM
judge, add relations, or repair a graph. Post-extraction semantic LLM calls are
forbidden and remain zero.

## Structural production gates

Every request must return HTTP 200, exactly one correctly named native tool
call, JSON arguments, a Pydantic-valid object, valid entity/relation indices,
evidence-backed fields and temporal bindings, no reasoning leakage, and no
length truncation. Any failure stops the run before embeddings and Neo4j.

FinalGraph validation rejects duplicate IDs, dangling endpoints, invalid
evidence, blank relations, invalid temporal references, and unresolved local
qualifier IDs. Projection acceptance requires exact JSON/Neo4j data, property,
schema, index, and namespace parity.

Evidence grounding is structural and provenance-addressable: accepted items
point to existing evidence spans, and temporal surfaces are checked against
their span. It does not mechanically prove entailment of every entity name,
predicate, condition, quantity, or version emitted by the model.

Run identity is independent of the output-directory basename. A validated or
generated run ID is bound with source, prompt, schema, contract, and provider
identity to form `graph_id`. Neo4j stores ownership metadata and replaces an
existing matching namespace in one managed data transaction; a different owner
is rejected before deletion. Authoritative JSON is persisted before projection
and remains available for a later rebuild if projection fails.

Neo4j structural label/type names are centralized in validated
`graph_projection` settings and recorded in `projection_manifest.json`. They
affect Cypher syntax, indexes, parity, rebuild, and Browser queries only; they do
not enter model-facing schemas or authoritative graph semantics.

## Development semantic benchmark

The public golden measures required relation/temporal recall and forbidden
relations. It is a report-only development benchmark for live artifacts and is
not imported by the production runner or included in `completion_status.json`.
Semantic omissions remain visible but do not invalidate a structurally valid
production graph.

Production guarantees structural integrity, not 100% fact recall, identical
raw-relation wording, or an identical decomposition of complex time intervals.

## Distribution boundary

The production package contains validated configuration, prompt assets,
`runtime/`, and the operational CLI modules in `scripts/`:
`run_production`, `open_graph`, and `rebuild_projection`. Development-only
model probes, public-golden evaluation, and project-conformance orchestration
live under `devtools/`. They remain tracked for review and CI but are excluded
from built wheels. Generated probe, conformance, golden, output, database, log,
and cache artifacts are ignored rather than packaged.
