---
name: text-processing-audit
description: >-
  Audit deterministic text processing for semantic regexes, keyword mappings,
  language-specific normalization, destructive rewriting, and hidden domain
  inference. Use when changing splitting, normalization, alias handling,
  comparison keys, or other preprocessing of source text. Allow representation
  normalization only when it cannot change meaning.
---

# Purpose

Ensure preprocessing remains deterministic, evidence-preserving, language-neutral, and free of hidden semantic inference.

## Use this skill when

- Changing splitting, normalization, aliases, or comparison keys.
- Auditing regexes, keyword maps, or source rewriting.

## Do not use this skill when

- Reviewing model request transport; use `llm-pipeline-review`.
- Reviewing graph endpoints after extraction; use `fact-graph-model-review`.

## Inputs

- Text-processing implementation, fixtures, offsets, and downstream consumers.
- Examples covering punctuation, versions, Unicode, whitespace, and boundaries.

## Workflow

1. Inventory every transformation and its consumer.
2. Classify representation normalization versus semantic inference.
3. Remove or reject hidden semantic logic.
4. Add focused boundary and preservation tests.
5. Run broader regression when units or identities can change.
6. Report transformations, evidence preservation, and risks.

## Required checks

- Preserve source text, evidence offsets, versions, quantities, polarity, and conditions.
- Reject semantic regexes, domain maps, language-specific normalizers, and sample patches.
- Run version, decimal, IP-address, and date splitting regressions.
- Verify evidence text and unit/document offset round trips exactly.
- Representation normalization may change formatting only when it cannot alter
  source meaning; it must not rewrite the authoritative source evidence.

## Output

Use the [review report template](../../templates/review-report.md) when a report is required.

## References

- Read the [text logic checklist](references/text-logic-checklist.md) when classifying transformations.
