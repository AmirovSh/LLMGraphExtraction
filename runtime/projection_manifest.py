"""Persisted binding between a run and its external projection type names."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.projection_identifiers import ProjectionTypeNames


PROJECTION_MANIFEST_VERSION = "1"


def projection_manifest(
    *, graph_id: str, projection: ProjectionTypeNames,
) -> dict[str, Any]:
    return {
        "manifest_version": PROJECTION_MANIFEST_VERSION,
        "graph_id": graph_id,
        "projection": projection.as_dict(),
    }


def load_projection_manifest(path: Path, *, graph_id: str) -> ProjectionTypeNames:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("projection manifest must contain an object")
    if payload.get("manifest_version") != PROJECTION_MANIFEST_VERSION:
        raise ValueError("unsupported projection manifest version")
    if payload.get("graph_id") != graph_id:
        raise ValueError("projection manifest graph_id differs from authoritative JSON")
    projection_payload = payload.get("projection")
    if not isinstance(projection_payload, dict):
        raise ValueError("projection manifest is missing projection metadata")
    return ProjectionTypeNames.from_manifest(projection_payload)


def require_matching_projection(
    *, recorded: ProjectionTypeNames, configured: ProjectionTypeNames,
) -> None:
    if recorded != configured:
        raise ValueError(
            "configured graph projection type names differ from the run's "
            "recorded projection_manifest.json"
        )
