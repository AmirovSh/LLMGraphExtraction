---
name: fact-graph-model-review
description: >-
  Review fact-graph semantics, entity boundaries, aliases, relation endpoints,
  qualifiers, evidence grounding, and JSON-to-Neo4j projection consistency.
  Use when changing entity resolution, relation extraction, graph construction,
  graph validation, or Neo4j projection. Do not use for general code style.
---

# Purpose

Verify that entities, relations, qualifiers, and evidence form a faithful graph without hidden semantic repair.

## Use this skill when

- Reviewing merges, aliases, endpoints, polarity, or qualifiers.
- Investigating graph-quality or JSON/Neo4j consistency defects.

## Do not use this skill when

- Reviewing general module boundaries; use `architecture-review`.
- Reviewing only persisted schema mechanics; use `schema-contract-review`.

## Inputs

- Authoritative JSON, source evidence, and relevant raw unit output.
- Resolution trace and projection/parity artifacts when applicable.

## Workflow

1. Trace each finding from source evidence through final JSON.
2. Classify extraction, normalization, resolution, or projection ownership.
3. Identify the owning stage and apply only an explicitly authorized correction.
4. Add focused semantic regression coverage.
5. Run full offline and clean end-to-end checks when semantics change.
6. Report evidence, parity, and residual limitations.

## Required checks

- Preserve evidence, stable IDs, qualifiers, and zero semantic patch calls.
- Verify JSON/Neo4j count, ID, endpoint, and graph namespace parity.
- Run behavioral negation and temporal/replacement regression tests.
- Run the synthetic-edge provenance test and inspect evidence lineage.
- When graph behavior changes, run
  `python -m devtools.check_public_golden --offline`; live semantic golden is a
  non-blocking development benchmark, while structural parity is a release gate.
- Do not attach the public-fixture golden checker to ordinary production
  completion; production graphs are final after their technical contracts pass.

## Output

Use the [review report template](../../templates/review-report.md) when a report is required.

## References

- Read the [fact-graph checklist](references/fact-graph-checklist.md) for semantic and projection review.
