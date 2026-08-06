---
name: llm-pipeline-review
description: >-
  Review LLM request construction, tool-call handling, provider profiles,
  retries, call accounting, prompt identity, and extraction-stage boundaries.
  Use when modifying model transport, structured output, prompt invocation,
  concurrency, retries, or LLM observability. Do not use for embedding-only or
  Neo4j-only changes.
---

# Purpose

Verify that extraction transport remains schema-constrained, observable, and limited to one successful call per unit.

## Use this skill when

- Changing request payloads, tool parsing, provider profiles, or timeouts.
- Reviewing retries, resume behavior, or call accounting.

## Do not use this skill when

- Diagnosing endpoint reachability only; use `local-model-connectivity`.
- Reviewing graph semantics after extraction; use `fact-graph-model-review`.

## Inputs

- Resolved configuration, request builder, transport adapter, and attempt artifacts.
- Offline transport fixtures and selected diagnostic evidence.

## Workflow

1. Inspect payload construction and response parsing.
2. Classify transport, provider compatibility, or semantic impact.
3. Apply the smallest explicitly authorized contract-preserving change.
4. Run focused offline transport and accounting tests.
5. Run endpoint diagnostics only when explicitly required.
6. Report attempts, usage, validation, and remaining risks.

## Required checks

- Successful extraction calls equal unit count; post-extraction semantic calls remain zero.
- Completed units are not reissued and secrets are redacted.

## Output

Use the [review report template](../../templates/review-report.md) when a report is required.

## References

- Read the [pipeline boundaries](references/pipeline-boundaries.md) for request and accounting checks.
