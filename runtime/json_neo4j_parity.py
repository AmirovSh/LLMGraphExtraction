"""Neo4j projection and exact parity checks for the authoritative JSON graph."""
from __future__ import annotations

import json
import os
from typing import Any

from neo4j import GraphDatabase

from config.projection_identifiers import DEFAULT_PROJECTION_TYPES, ProjectionTypeNames
from config.settings import ProjectSettings, environment_value

EXPECTED_NODE_LABEL = str(DEFAULT_PROJECTION_TYPES.entity_label)
EXPECTED_RELATIONSHIP_TYPE = str(DEFAULT_PROJECTION_TYPES.relation_type)
EXPECTED_PROJECT_INDEXES = DEFAULT_PROJECTION_TYPES.expected_indexes
REQUIRED_NODE_PROPERTIES = {
    "entity_id", "graph_id", "canonical_name", "primary_type", "aliases",
    "local_entity_ids", "evidence_span_ids", "display_name", "run_id",
    "source_sha256", "manifest_identity_hash",
}
REQUIRED_RELATIONSHIP_PROPERTIES = {
    "edge_id", "graph_id", "source_entity_id", "target_entity_id", "raw_relation",
    "relation_family", "relation_description", "evidence_span_ids", "support_level",
    "unit_id", "display_relation", "qualifiers_json", "provenance_json",
    "run_id", "source_sha256", "manifest_identity_hash",
}


class NamespaceOwnershipError(RuntimeError):
    """Raised before replacement when an existing namespace has another owner."""


def neo4j_settings(settings: ProjectSettings) -> dict[str, str]:
    config = settings.neo4j
    return {
        "uri": environment_value(config.uri_env),
        "username": environment_value(config.username_env),
        "password": environment_value(config.password_env),
        "database": os.environ.get(config.database_env, config.default_database),
    }


def _node_projection_properties(
    node: dict[str, Any], graph_id: str, ownership: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        **node, "graph_id": graph_id, "display_name": node["canonical_name"],
        **(ownership or {}),
    }


def neo4j_relationship_properties(
    edge: dict[str, Any], graph_id: str, ownership: dict[str, str] | None = None,
) -> dict[str, Any]:
    validate_relationship_raw_relation(edge)
    props = {key: value for key, value in edge.items() if key not in {"qualifiers", "provenance"}}
    props.update({
        "graph_id": graph_id,
        "display_relation": edge["raw_relation"],
        "qualifiers_json": json.dumps(edge["qualifiers"], ensure_ascii=False, sort_keys=True),
        "provenance_json": json.dumps(edge["provenance"], ensure_ascii=False, sort_keys=True),
        **(ownership or {}),
    })
    return props


def _property_mismatches(
    expected: dict[str, dict[str, Any]], actual: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for item_id in sorted(set(expected) & set(actual)):
        for property_name, expected_value in expected[item_id].items():
            actual_value = actual[item_id].get(property_name)
            if actual_value != expected_value:
                mismatches.append({
                    "id": item_id,
                    "property": property_name,
                    "json_value": expected_value,
                    "neo4j_value": actual_value,
                })
    return mismatches


def build_parity_diff(
    *, graph_id: str, json_nodes: list[dict[str, Any]], json_edges: list[dict[str, Any]],
    neo4j_node_ids: set[str], neo4j_edges: dict[str, tuple[str, str]],
    edges_without_graph_id: list[dict[str, Any]], duplicate_edges: list[dict[str, Any]],
    import_errors: list[str], neo4j_node_properties: dict[str, dict[str, Any]] | None = None,
    neo4j_edge_properties: dict[str, dict[str, Any]] | None = None,
    active_project_labels: set[str] | None = None,
    active_project_relationship_types: set[str] | None = None,
    project_indexes: list[dict[str, Any]] | None = None,
    unexpected_graph_ids: list[str] | None = None,
    ownership: dict[str, str] | None = None,
    projection: ProjectionTypeNames = DEFAULT_PROJECTION_TYPES,
) -> dict[str, Any]:
    json_node_ids = {item["entity_id"] for item in json_nodes}
    json_edge_ids = {item["edge_id"] for item in json_edges}
    source_target_mismatches = [
        {
            "edge_id": edge["edge_id"],
            "json_source": edge["source_entity_id"],
            "json_target": edge["target_entity_id"],
            "neo4j_source": neo4j_edges[edge["edge_id"]][0],
            "neo4j_target": neo4j_edges[edge["edge_id"]][1],
        }
        for edge in json_edges
        if edge["edge_id"] in neo4j_edges
        and neo4j_edges[edge["edge_id"]]
        != (edge["source_entity_id"], edge["target_entity_id"])
    ]
    data_parity = {
        "json_node_count": len(json_node_ids),
        "neo4j_node_count": len(neo4j_node_ids),
        "json_edge_count": len(json_edge_ids),
        "neo4j_edge_count": len(neo4j_edges),
        "json_only_nodes": sorted(json_node_ids - neo4j_node_ids),
        "neo4j_only_nodes": sorted(neo4j_node_ids - json_node_ids),
        "json_only_edges": sorted(json_edge_ids - set(neo4j_edges)),
        "neo4j_only_edges": sorted(set(neo4j_edges) - json_edge_ids),
        "source_target_mismatches": source_target_mismatches,
        "duplicate_edges": [item["edge_id"] for item in duplicate_edges],
        "import_errors": import_errors,
    }

    node_properties = neo4j_node_properties or {}
    edge_properties = neo4j_edge_properties or {}
    expected_node_properties = (
        {node["entity_id"]: _node_projection_properties(node, graph_id, ownership) for node in json_nodes}
        if neo4j_node_properties is not None else {}
    )
    expected_edge_properties = (
        {
            edge["edge_id"]: neo4j_relationship_properties(edge, graph_id, ownership)
            for edge in json_edges
        }
        if neo4j_edge_properties is not None else {}
    )
    missing_node_properties = {
        node_id: sorted(REQUIRED_NODE_PROPERTIES - set(properties))
        for node_id, properties in node_properties.items()
        if REQUIRED_NODE_PROPERTIES - set(properties)
    }
    missing_relationship_properties = {
        edge_id: sorted(REQUIRED_RELATIONSHIP_PROPERTIES - set(properties))
        for edge_id, properties in edge_properties.items()
        if REQUIRED_RELATIONSHIP_PROPERTIES - set(properties)
    }
    expected_node_label = str(projection.entity_label)
    expected_relationship_type = str(projection.relation_type)
    expected_indexes = projection.expected_indexes
    labels = active_project_labels if active_project_labels is not None else {expected_node_label}
    relationship_types = (
        active_project_relationship_types
        if active_project_relationship_types is not None
        else {expected_relationship_type}
    )
    indexes = project_indexes
    index_names = {item.get("name") for item in indexes or []}
    unexpected_indexes = []
    misconfigured_indexes = []
    expected_index_targets = {
        projection.entity_index: expected_node_label,
        projection.relation_index: expected_relationship_type,
    }
    if indexes is not None:
        for item in indexes:
            labels_or_types = set(item.get("labelsOrTypes") or [])
            if labels_or_types & {expected_node_label, expected_relationship_type}:
                if item.get("name") not in expected_indexes:
                    unexpected_indexes.append(item.get("name"))
            expected_target = expected_index_targets.get(item.get("name"))
            if expected_target is not None and labels_or_types != {expected_target}:
                misconfigured_indexes.append(item.get("name"))
            properties = item.get("properties")
            if (
                expected_target is not None
                and properties is not None
                and list(properties) != ["graph_id"]
            ):
                misconfigured_indexes.append(item.get("name"))
    schema_parity = {
        "expected_node_labels": [expected_node_label],
        "active_project_labels": sorted(labels),
        "missing_expected_node_labels": sorted({expected_node_label} - labels),
        "unexpected_project_labels": sorted(labels - {expected_node_label}),
        "expected_relationship_types": [expected_relationship_type],
        "active_project_relationship_types": sorted(relationship_types),
        "missing_expected_relationship_types": sorted(
            {expected_relationship_type} - relationship_types
        ),
        "unexpected_project_relationship_types": sorted(
            relationship_types - {expected_relationship_type}
        ),
        "missing_required_node_properties": missing_node_properties,
        "missing_required_relationship_properties": missing_relationship_properties,
        "node_property_mismatches": _property_mismatches(expected_node_properties, node_properties),
        "relationship_property_mismatches": _property_mismatches(
            expected_edge_properties, edge_properties
        ),
        "missing_project_indexes": (
            sorted(expected_indexes - index_names) if indexes is not None else []
        ),
        "unexpected_project_indexes": sorted(
            item for item in unexpected_indexes if isinstance(item, str)
        ),
        "misconfigured_project_indexes": sorted(set(
            item for item in misconfigured_indexes if isinstance(item, str)
        )),
    }
    namespace_parity = {
        "edges_without_graph_id": edges_without_graph_id,
        "unexpected_graph_ids": sorted(unexpected_graph_ids or []),
    }
    failed_values = [
        *(
            data_parity[key]
            for key in (
                "json_only_nodes", "neo4j_only_nodes", "json_only_edges",
                "neo4j_only_edges", "source_target_mismatches", "duplicate_edges",
                "import_errors",
            )
        ),
        *(
            schema_parity[key]
            for key in (
                "missing_expected_node_labels", "unexpected_project_labels",
                "missing_expected_relationship_types", "unexpected_project_relationship_types",
                "missing_required_node_properties", "missing_required_relationship_properties",
                "node_property_mismatches", "relationship_property_mismatches",
                "missing_project_indexes", "unexpected_project_indexes",
                "misconfigured_project_indexes",
            )
        ),
        namespace_parity["edges_without_graph_id"],
        namespace_parity["unexpected_graph_ids"],
    ]
    status = "failed" if any(failed_values) else "passed"
    return {
        "graph_id": graph_id,
        "data_parity": data_parity,
        "schema_parity": schema_parity,
        "projection": projection.as_dict(),
        "namespace_parity": namespace_parity,
        "status": status,
        **data_parity,
        **namespace_parity,
    }


def validate_relationship_raw_relation(edge: dict[str, Any]) -> None:
    raw_relation = edge.get("raw_relation")
    if not isinstance(raw_relation, str) or not raw_relation.strip():
        edge_id = edge.get("edge_id", "<unknown>")
        raise ValueError(f"{edge_id}: raw_relation must be a non-empty string")


def _index_record(record: Any) -> dict[str, Any]:
    return {
        "name": record["name"],
        "type": record["type"],
        "labelsOrTypes": list(record["labelsOrTypes"] or []),
        "properties": list(record["properties"] or []),
    }


def replace_namespace_transactionally(
    session: Any, *, graph_id: str, nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]], ownership: dict[str, str],
    projection: ProjectionTypeNames = DEFAULT_PROJECTION_TYPES,
) -> None:
    """Replace one owned namespace in a single managed write transaction."""
    expected_owner = {
        key: ownership[key]
        for key in ("run_id", "source_sha256", "manifest_identity_hash")
    }
    entity_label = projection.entity_label.cypher
    relation_type = projection.relation_type.cypher

    def replace(tx: Any) -> None:
        existing = list(tx.run(
            "MATCH (n {graph_id:$graph_id}) "
            "OPTIONAL MATCH (n)-[r {graph_id:$graph_id}]-() "
            "RETURN DISTINCT n.run_id AS run_id, n.source_sha256 AS source_sha256, "
            "n.manifest_identity_hash AS manifest_identity_hash, labels(n) AS labels, "
            "collect(DISTINCT type(r)) AS relationship_types LIMIT 2",
            graph_id=graph_id,
        ))
        for record in existing:
            actual = {key: record[key] for key in expected_owner}
            if actual != expected_owner:
                raise NamespaceOwnershipError(
                    f"Neo4j namespace {graph_id!r} belongs to a different run identity"
                )
            labels = set(record.get("labels") or [entity_label])
            relationship_types = set(record.get("relationship_types") or [relation_type])
            if labels != {entity_label} or relationship_types - {relation_type}:
                raise NamespaceOwnershipError(
                    f"Neo4j namespace {graph_id!r} uses different projection type names"
                )
        tx.run(
            "MATCH (n {graph_id:$graph_id}) DETACH DELETE n",
            graph_id=graph_id,
        ).consume()
        node_rows = [
            {
                "entity_id": node["entity_id"],
                "props": _node_projection_properties(node, graph_id, expected_owner),
            }
            for node in nodes
        ]
        tx.run(
            "UNWIND $rows AS row "
            f"CREATE (n:{entity_label}) SET n = row.props",
            rows=node_rows,
        ).consume()
        edge_rows = [
            {
                "source": edge["source_entity_id"],
                "target": edge["target_entity_id"],
                "props": neo4j_relationship_properties(edge, graph_id, expected_owner),
            }
            for edge in edges
        ]
        tx.run(
            "UNWIND $rows AS row "
            f"MATCH (a:{entity_label} {{graph_id:$graph_id,entity_id:row.source}}) "
            f"MATCH (b:{entity_label} {{graph_id:$graph_id,entity_id:row.target}}) "
            f"CREATE (a)-[r:{relation_type}]->(b) SET r = row.props",
            graph_id=graph_id, rows=edge_rows,
        ).consume()

    session.execute_write(replace)


def import_neo4j(
    graph_id: str, nodes: list[dict[str, Any]], edges: list[dict[str, Any]],
    settings: ProjectSettings, ownership: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    for edge in edges:
        validate_relationship_raw_relation(edge)
    connection = neo4j_settings(settings)
    projection = settings.graph_projection.type_names
    entity_label = projection.entity_label.cypher
    relation_type = projection.relation_type.cypher
    entity_index = projection.entity_index
    relation_index = projection.relation_index
    node_ids = {node["entity_id"] for node in nodes}
    import_errors = []
    for edge in edges:
        if edge["source_entity_id"] not in node_ids or edge["target_entity_id"] not in node_ids:
            import_errors.append(f"{edge['edge_id']}: endpoint is absent from JSON nodes")
    with GraphDatabase.driver(
        connection["uri"], auth=(connection["username"], connection["password"])
    ) as driver:
        with driver.session(database=connection["database"]) as session:
            session.run(
                f"CREATE INDEX {entity_index} IF NOT EXISTS "
                f"FOR (n:{entity_label}) ON (n.graph_id)"
            ).consume()
            session.run(
                f"CREATE INDEX {relation_index} IF NOT EXISTS "
                f"FOR ()-[r:{relation_type}]-() ON (r.graph_id)"
            ).consume()
            if not import_errors:
                replace_namespace_transactionally(
                    session, graph_id=graph_id, nodes=nodes, edges=edges,
                    ownership=ownership,
                    projection=projection,
                )
            node_records = list(session.run(
                f"MATCH (n:{entity_label} {{graph_id:$graph_id}}) "
                "RETURN n.entity_id AS entity_id, properties(n) AS properties",
                graph_id=graph_id,
            ))
            edge_records = list(session.run(
                f"MATCH (a:{entity_label})-[r:{relation_type} {{graph_id:$graph_id}}]->(b:{entity_label}) "
                "RETURN r.edge_id AS edge_id, a.entity_id AS source, b.entity_id AS target, "
                "properties(r) AS properties",
                graph_id=graph_id,
            ))
            no_graph_id = [dict(record) for record in session.run(
                f"MATCH ()-[r:{relation_type}]->() WHERE r.graph_id IS NULL "
                "RETURN elementId(r) AS relationship_id, type(r) AS relationship_type"
            )]
            active_labels = {
                label for record in session.run(
                    "MATCH (n {graph_id:$graph_id}) "
                    "UNWIND labels(n) AS label RETURN DISTINCT label",
                    graph_id=graph_id,
                ) for label in [record["label"]]
            }
            active_types = {
                record["relationship_type"] for record in session.run(
                    "MATCH ()-[r {graph_id:$graph_id}]->() "
                    "RETURN DISTINCT type(r) AS relationship_type",
                    graph_id=graph_id,
                )
            }
            indexes = [_index_record(record) for record in session.run(
                "SHOW INDEXES YIELD name, type, labelsOrTypes, properties "
                "RETURN name, type, labelsOrTypes, properties"
            )]
            invalid_graph_ids = [
                str(record["graph_id"]) for record in session.run(
                    "MATCH (n) WHERE n.graph_id IS NOT NULL "
                    "AND trim(toString(n.graph_id)) = '' "
                    "RETURN DISTINCT n.graph_id AS graph_id "
                    "UNION MATCH ()-[r]->() WHERE r.graph_id IS NOT NULL "
                    "AND trim(toString(r.graph_id)) = '' "
                    "RETURN DISTINCT r.graph_id AS graph_id"
                )
            ]
    neo4j_node_ids = {record["entity_id"] for record in node_records}
    neo4j_edges = {
        record["edge_id"]: (record["source"], record["target"]) for record in edge_records
    }
    diff = build_parity_diff(
        graph_id=graph_id,
        json_nodes=nodes,
        json_edges=edges,
        neo4j_node_ids=neo4j_node_ids,
        neo4j_edges=neo4j_edges,
        edges_without_graph_id=no_graph_id,
        duplicate_edges=[],
        import_errors=import_errors,
        neo4j_node_properties={
            record["entity_id"]: dict(record["properties"]) for record in node_records
        },
        neo4j_edge_properties={
            record["edge_id"]: dict(record["properties"]) for record in edge_records
        },
        active_project_labels=active_labels,
        active_project_relationship_types=active_types,
        project_indexes=indexes,
        unexpected_graph_ids=invalid_graph_ids,
        ownership=ownership,
        projection=projection,
    )
    return {
        "graph_id": graph_id,
        "nodes": len(neo4j_node_ids),
        "edges": len(neo4j_edges),
        "projection": projection.as_dict(),
    }, diff
