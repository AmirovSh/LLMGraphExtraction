# Coding Agent Entry Point

This repository builds evidence-backed fact graphs from formal documents:

`text -> deterministic evidence spans -> one extraction call per span -> validation -> bge-m3 -> observable resolution -> deterministic JSON graph -> Neo4j projection`

## Always-on rules

- JSON is authoritative; Neo4j is an exact derived projection.
- Permit one successful extraction LLM call per deterministic evidence-span unit and zero post-extraction semantic LLM calls.
- Preserve evidence grounding, `canonical_name_only` embeddings, observable resolution, stable IDs, and truthful call accounting.
- Do not introduce semantic regexes, domain keyword maps, language-specific semantic normalizers, or sample-specific runtime patches.
- Treat prompt/schema meaning, extraction behavior, entity resolution, graph semantics, and projection contracts as semantic changes. Require focused regression coverage and a clean end-to-end run before accepting them.
- Verify repository root, branch, status, and relevant tests before editing. Run focused checks, then `python -m pytest` and `git diff --check`.
- Never commit secrets, credentials, runtime outputs, raw model responses, databases, dumps, caches, or local audit reports.
- Do not configure a remote, push, or modify protected release references without explicit authorization.
- Golden validation is a development/release regression tool, never production
  post-processing for arbitrary user text.
- A production graph is final after extraction, validation, consolidation,
  persistence, and configured projection parity complete successfully.

Read supporting policies only when relevant:

- [Architecture](.agents/policies/architecture.md)
- [Security and artifacts](.agents/policies/security-and-artifacts.md)
- [Text processing](.agents/policies/text-processing.md)
- [Model connections](.agents/policies/model-connections.md)
- [Long-running work](.agents/policies/long-running-work.md)

Use the matching task-specific skill under [.agents/skills](.agents/README.md) for specialized reviews and diagnostics.
