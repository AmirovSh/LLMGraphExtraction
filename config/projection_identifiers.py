"""Validated structural identifiers for Neo4j and public graph views."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_IDENTIFIER_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


class ProjectionIdentifier(str):
    """A Cypher identifier safe to render without quoting or normalization."""

    def __new__(cls, value: str) -> "ProjectionIdentifier":
        if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
            raise ValueError(
                "projection identifier must match ^[A-Z][A-Z0-9_]{0,63}$: "
                f"{value!r}"
            )
        return str.__new__(cls, value)

    @property
    def cypher(self) -> str:
        return str(self)


@dataclass(frozen=True)
class ProjectionTypeNames:
    entity_label: ProjectionIdentifier
    relation_type: ProjectionIdentifier

    @classmethod
    def parse(cls, *, entity_label: str, relation_type: str) -> "ProjectionTypeNames":
        return cls(
            entity_label=ProjectionIdentifier(entity_label),
            relation_type=ProjectionIdentifier(relation_type),
        )

    @classmethod
    def from_manifest(cls, payload: dict[str, Any]) -> "ProjectionTypeNames":
        if set(payload) != {"entity_label", "relation_type"}:
            raise ValueError("projection metadata must contain only entity_label and relation_type")
        return cls.parse(
            entity_label=payload["entity_label"],
            relation_type=payload["relation_type"],
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "entity_label": str(self.entity_label),
            "relation_type": str(self.relation_type),
        }

    @property
    def expected_indexes(self) -> set[str]:
        return {self.entity_index, self.relation_index}

    @property
    def entity_index(self) -> str:
        if self.entity_label == "FACT_ENTITY":
            return "fact_entity_graph_id"
        return f"{str(self.entity_label).lower()}_node_graph_id"

    @property
    def relation_index(self) -> str:
        if self.relation_type == "FACT_RELATION":
            return "fact_relation_graph_id"
        return f"{str(self.relation_type).lower()}_relationship_graph_id"


DEFAULT_PROJECTION_TYPES = ProjectionTypeNames.parse(
    entity_label="FACT_ENTITY",
    relation_type="FACT_RELATION",
)
