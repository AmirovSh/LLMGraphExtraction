---
name: schema-contract-review
description: >-
  Review JSON schemas, Pydantic models, manifests, artifact contracts, ID
  invariants, required fields, and JSON-to-Neo4j parity. Use when adding,
  removing, renaming, validating, or serializing persisted fields and artifacts.
  Do not use for purely visual Neo4j Browser styling.
---

# Purpose

Verify that persisted contracts are explicit, validated, deterministic, compatible, and projected without loss.

## Use this skill when

- Adding, removing, renaming, or validating persisted fields.
- Changing manifests, IDs, serialization, artifacts, or parity checks.

## Do not use this skill when

- Reviewing relation meaning; use `fact-graph-model-review`.
- Changing only Browser captions or colors.

## Inputs

- Schema/model definitions, example artifacts, serializer, and consumers.
- Migration/compatibility requirements and parity tests.

## Workflow

1. Trace each field from producer through validation and consumers.
2. Classify active, static, removed, or unsupported contract behavior.
3. Apply the smallest explicitly authorized contract change.
4. Add invalid-input and round-trip tests.
5. Run full offline and parity checks when persisted behavior changes.
6. Report compatibility, migration, and remaining risks.

## Required checks

- Validate required fields before external writes.
- Preserve deterministic IDs, redaction, endpoint integrity, and parity.
- Validate `FinalGraph` before persisting `fact_graph.json`.
- Run required-field, ID-integrity, and endpoint failure tests.
- Run data, schema, and namespace parity checks; reject unexpected Neo4j
  project labels, relationship types, and indexes.

## Output

Use the [review report template](../../templates/review-report.md) when a report is required.

## References

- Read the [schema checklist](references/schema-checklist.md) for persisted-contract review.
