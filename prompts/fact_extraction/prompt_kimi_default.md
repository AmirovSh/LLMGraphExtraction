You receive exactly one evidence span. Return exactly one native tool call and
no prose, reasoning, analysis, markdown, or XML.

EVIDENCE SPAN
{allowed_evidence_spans}

Extract every explicit graphable factual predicate from this span.

For each predicate:

- preserve its explicit subject;
- preserve its direct semantic object;
- create entities for common-noun arguments when necessary;
- keep locations, beneficiaries, repositories, and other context separate;
- preserve essential verb complements in `raw_relation`;
- copy explicit temporal, conditional, quantitative, modal, version, and
  negation qualifiers;
- create one relation for each explicit graphable predicate.

A direct object must not be replaced by a nearby location or beneficiary. A
common-noun phrase may be a graph entity.

For `Reader reads decisions from Registry`, use the reader as source,
`reads decisions from` as `raw_relation`, and Registry as target. For `Portal
displays evidence`, use the portal as source, `displays` as `raw_relation`, and
evidence as target. Explicit temporal wording must be returned in a temporal
binding for the relation it modifies. Set `negated=true` only when the
proposition itself is denied; positive blocks, prevents, rejects, and disables
predicates are not negated.

Use `facts_present` when at least one binary relation is extracted. Use
`no_graph_fact` with one allowed factual reason only when this span contains no
graphable binary fact.

Before submitting the tool call:

1. Verify every explicit predicate in the single span.
2. Verify every predicate has its correct source and target.
3. Verify no contextual entity replaced a direct object.
4. Verify common-noun arguments were not omitted.
5. Verify essential predicate complements were preserved.
6. Verify every explicit qualifier was copied.
7. Verify every relation endpoint exists in entities.
8. Return exactly one valid native tool call.

Return semantic entities and relations only. Use zero-based entity indices.
Use zero-based relation indices in `temporal_bindings`. Always return
`temporal_bindings`; use `[]` when no relation has explicit time. Do not create
IDs or evidence references. `relation_family` is required for every relation.

SOFT ENTITY GUIDANCE
{entity_guidance}

SOFT RELATION FAMILY GUIDANCE
{relation_family_guidance}
