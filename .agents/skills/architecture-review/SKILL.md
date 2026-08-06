---
name: architecture-review
description: >-
  Review repository architecture, module boundaries, dependency direction,
  orchestration responsibilities, and maintainability risks. Use when changing
  package structure, splitting large modules, introducing adapters, or reviewing
  whether a refactor preserves the project's architectural invariants. Do not
  use for graph-semantic quality review.
---

# Purpose

Assess whether a structural change preserves clear ownership, dependency direction, and production behavior.

## Use this skill when

- Splitting or moving runtime modules.
- Introducing adapters or changing orchestration boundaries.

## Do not use this skill when

- Reviewing entity or relation meaning; use `fact-graph-model-review`.
- Reviewing persisted fields; use `schema-contract-review`.

## Inputs

- Proposed diff and reachable production call path.
- Protecting tests and relevant configuration.

## Workflow

1. Inspect entry points, imports, ownership, and state flow.
2. Classify structural versus semantic effects.
3. Apply the smallest explicitly authorized boundary change.
4. Run focused characterization tests.
5. Run broader checks when production reachability changes.
6. Report evidence, coupling, and remaining risks.

## Required checks

- Confirm no new model-call or semantic stage.
- Run `python -m pytest` and `git diff --check` before handoff.

## Output

Use the [review report template](../../templates/review-report.md) when a report is required.

## References

- Read [architecture invariants](references/architecture-invariants.md) when classifying boundaries and dependencies.
