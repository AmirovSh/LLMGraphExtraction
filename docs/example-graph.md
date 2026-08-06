# Neo4j graph example

This example uses the reviewed synthetic source in
[`examples/sample_input.txt`](../examples/sample_input.txt). The graph must be
created by the normal extraction pipeline, validated as authoritative
`fact_graph.json`, projected through the existing Neo4j importer, and accepted
by JSON/Neo4j parity. It is not separately reconstructed for documentation.

The documented run ID is `neo4j_public_example`. With the current protected
prompt, schema, source, and provider identity, its graph namespace is
`fact_extraction_neo4j_public_example_107f73711793`. Confirm the value in the
run's `fact_graph.json` and `projection_manifest.json` before querying; an
intentional identity change will produce a different suffix.

## Example visualization

![Example Neo4j fact graph](images/visualisation.svg)

This SVG is a static example of the graph displayed from the Neo4j projection.
It helps illustrate the graph's overall shape, but it is not an authoritative
artifact and is not a separate graph reconstruction. The validated
`fact_graph.json` remains authoritative; the live Neo4j namespace is its
rebuildable derived projection.

> [!NOTE]
> The visualization is an example output rather than a semantic-quality
> baseline. Extraction is probabilistic, so another accepted run may differ
> while still satisfying the schema, evidence, call-accounting, and projection
> contracts.

## Read-only Browser query

Set the exact namespace in Neo4j Browser, then return only real projected nodes
and relationships owned by that namespace:

```cypher
:param graph_id => 'fact_extraction_neo4j_public_example_107f73711793';
```

```cypher
MATCH (source:FACT_ENTITY)-[relation:FACT_RELATION]->(target:FACT_ENTITY)
WHERE source.graph_id = $graph_id
  AND relation.graph_id = $graph_id
  AND target.graph_id = $graph_id
RETURN source, relation, target
```

If parity reports evidence-backed isolated nodes, display them separately:

```cypher
MATCH (node:FACT_ENTITY {graph_id: $graph_id})
WHERE NOT (node)-[:FACT_RELATION {graph_id: $graph_id}]-()
RETURN node
```

For custom configured projection names, replace only the structural
`FACT_ENTITY` label and `FACT_RELATION` relationship type with the exact values
recorded in `projection_manifest.json`. Keep `graph_id` filtering unchanged.

## Browser captions and style

The tracked [`config/neo4j_browser.grass`](../config/neo4j_browser.grass)
stylesheet uses:

- node caption: `canonical_name`;
- relationship caption: `raw_relation`.

In Browser, run `:style` and upload that stylesheet. The structural label and
type remain `FACT_ENTITY` and `FACT_RELATION`, while `primary_type` preserves
the semantic node type and `relation_family`, `relation_description`,
`raw_relation`, evidence IDs, qualifier JSON, and provenance JSON remain
relationship properties. Meaningful predicates such as `coordinates`,
`reads decisions from`, `applies`, and `displays` therefore remain visible as
edge captions.

The default stylesheet targets the default projection names. When using custom
names, copy the two style selectors locally and replace only their label/type;
do not commit environment-specific Browser styling.

## Refreshing the visualization

The checked-in SVG can be refreshed from a reviewed Neo4j Browser projection:

1. Copy `.env.neo4j.example` to ignored `.env.neo4j`, set local credentials,
   and start `docker compose --env-file .env.neo4j -f docker-compose.neo4j.yml up -d`.
2. Run the normal pipeline with `--input examples/sample_input.txt --run-id neo4j_public_example`.
   Use configured model and embedding endpoints only in the owner's controlled
   acceptance environment; resume is not supported.
3. Confirm `completion_status.json` and `json_neo4j_edge_diff.json` report
   successful projection/parity, and copy the exact `graph_id` from
   `fact_graph.json`.
4. Run `python -m scripts.open_graph --run-id neo4j_public_example --open --show-query`.
5. In Neo4j Browser, select the configured database, run the namespace-filtered
   query above, and apply `config/neo4j_browser.grass` through `:style`.
6. Use only Neo4j Browser layout controls, fit the graph to the viewport, and
   hide the property panel unless it is needed to explain one reviewed field.
7. Capture only the graph canvas. Exclude bookmarks, connection URLs,
   credentials, local paths, unrelated namespaces, browser chrome containing
   personal information, and sensitive source/debug panels.
8. Export the reviewed graph as `docs/images/visualisation.svg`, verify that the
   SVG has no scripts, external resources, credentials, local paths, or
   unrelated graph namespaces, and review the documentation diff before
   committing it.

The displayed graph is the actual Neo4j projection. JSON remains authoritative;
Neo4j is its verified, rebuildable derived projection. Semantic extraction is
probabilistic even when projection and parity are deterministic for an accepted
graph.
