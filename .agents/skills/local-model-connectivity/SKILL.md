---
name: local-model-connectivity
description: >-
  Diagnose local OpenAI-compatible LLM and embedding endpoints, model discovery,
  credentials presence, tool-call compatibility, and connectivity failures.
  Use when a local model, embedding service, proxy, or provider profile cannot
  be reached or returns transport errors. Never print or persist secret values.
---

# Purpose

Diagnose local model infrastructure without changing extraction semantics or exposing credentials.

## Use this skill when

- An LLM or embedding endpoint is unreachable, slow, or incompatible.
- Model discovery, tool calls, or provider-profile transport fails.

## Do not use this skill when

- Changing prompt meaning; use `llm-pipeline-review`.
- Reviewing embedding-based merge decisions; use `fact-graph-model-review`.

## Inputs

- Presence-only environment checks and resolved non-secret configuration.
- Sanitized status, timing, exception, and server-log evidence.

## Workflow

1. Verify configuration presence without printing values.
2. Separate client, network, server, and compatibility failures.
3. Run the smallest explicit connectivity probe.
4. Avoid writes and semantic changes.
5. Run offline tests for diagnostic code.
6. Report sanitized evidence and the next infrastructure action.

## Required checks

- Never print credentials, headers, or private payloads.
- Do not add retries, model switching, or prompt changes during diagnosis.

## Output

Use the [review report template](../../templates/review-report.md) when a report is required.

## References

- Read the [connectivity checklist](references/connectivity-checklist.md) before a live probe.
