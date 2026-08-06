---
name: regression-golden-review
description: >-
  Run and review deterministic regression, public sample, and clean end-to-end
  graph tests. Use when semantic behavior, prompt wording, entity resolution,
  graph projection, or persisted artifacts may have changed and must be compared
  with an accepted baseline. Do not update expected results merely to make tests pass.
---

# Purpose

Establish behavior-preservation or intentional semantic change with reviewed fixtures and clean-run evidence.
This is a development/release regression skill; it is not a production
post-processing stage for arbitrary user runs.

## Use this skill when

- Prompt, resolution, graph, projection, or persisted output may change.
- A golden fixture or clean end-to-end result must be reviewed.

## Do not use this skill when

- Only code formatting or documentation changes.
- Connectivity is the sole failure; use `local-model-connectivity`.

## Inputs

- Accepted fixture/baseline and proposed output.
- Call accounting, validation, resolution, parity, and source evidence.

## Workflow

1. Identify the accepted baseline and protected invariants.
2. Run focused deterministic regression.
3. Classify every difference before updating expectations.
4. Run full offline tests.
5. Run a clean end-to-end acceptance only when required.
6. Report metrics, automated semantic assertion results, parity, and risks.

## Required checks

- Never update expected output solely to pass.
- Verify truthful calls, evidence, stable IDs, and JSON/Neo4j parity.
- Normal validation must never mutate the golden manifest.
- Use a new no-resume namespace and distinguish offline characterization from
  live model acceptance.
- Run `python -m devtools.check_public_golden --offline` for the deterministic
  gate; use `--live` only with an explicitly available environment.
- Semantic golden validation is an independent report-only quality benchmark;
  it cannot make structural parity pass or fail.

## Output

Use the [review report template](../../templates/review-report.md) when a report is required.

## References

- Read the [golden review checklist](references/golden-review-checklist.md) before accepting output changes.
