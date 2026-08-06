# Architecture Policy

## Runtime boundary

- `scripts.run_production` is the supported production entry point.
- Centralized YAML and strict settings models own active mutable configuration.
- Deterministic splitting precedes one extraction call per evidence-span unit.
- Schema and evidence validation precede embeddings and entity resolution.
- Exact consolidation creates the authoritative JSON graph.
- Neo4j is rebuilt from JSON as a query and visualization projection.

## Semantic invariants

- Primary successful calls equal unit count.
- Post-extraction semantic calls, merge verifiers, fact-layer calls, gleaning, and semantic patches remain zero.
- Embedding input remains `canonical_name_only`.
- Entity type is trace-only, never a hard merge filter.
- Resolution thresholds stay centrally configured and decisions remain observable.
- Qualifier references are rebound before serialization; unresolved local IDs are forbidden.
- Evidence and provenance remain attached to accepted relations.

Keep orchestration, transport, validation, resolution, persistence, and projection responsibilities separate. Do not add hidden fallbacks or sample-specific branches. Semantic changes require focused regression coverage, full offline tests, and a clean end-to-end run with call accounting and JSON/Neo4j parity.
