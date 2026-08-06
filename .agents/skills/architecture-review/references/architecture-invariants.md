# Architecture Review Checklist

- Trace from the supported entry point rather than unused modules.
- Keep transport, validation, resolution, persistence, and projection ownership distinct.
- Keep active configuration centralized and validated.
- Preserve authoritative JSON and exact Neo4j projection.
- Identify tests that protect every moved responsibility.
- Flag circular imports, hidden global state, duplicate defaults, and unreachable compatibility paths.
