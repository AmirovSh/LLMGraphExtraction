"""Collision-safe, secret-free production run and source identity."""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from runtime.artifact_store import sha256_json, sha256_text

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def validate_run_id(value: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "run_id must be 1-64 characters using letters, digits, dot, underscore, or hyphen"
        )
    return value


def generate_run_id() -> str:
    return f"run_{uuid.uuid4().hex}"


def source_identity(input_path: Path, source: str, repository_root: Path) -> dict[str, Any]:
    resolved = input_path.resolve(strict=True)
    metadata: dict[str, Any] = {
        "source_name": resolved.name,
        "source_sha256": sha256_text(source),
        "source_size_bytes": len(source.encode("utf-8")),
        "source_location_policy": "external",
        "repository_relative_path": None,
    }
    try:
        relative = resolved.relative_to(repository_root.resolve(strict=True))
    except ValueError:
        return metadata
    metadata["source_location_policy"] = "repository_relative"
    metadata["repository_relative_path"] = relative.as_posix()
    return metadata


def build_run_identity(
    *, run_id: str, source: dict[str, Any], prompt_hash: str, schema_hash: str,
    contract_id: str, provider_profile_id: str,
) -> dict[str, str]:
    fields = {
        "run_id": validate_run_id(run_id),
        "source_sha256": str(source["source_sha256"]),
        "prompt_hash": prompt_hash,
        "schema_hash": schema_hash,
        "contract_id": contract_id,
        "provider_profile_id": provider_profile_id,
    }
    identity_hash = sha256_json(fields)
    return {
        **fields,
        "manifest_identity_hash": identity_hash,
        "graph_id": f"fact_extraction_{run_id}_{identity_hash[:12]}",
    }
