"""Atomic artifact persistence and centralized production-run paths."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_text(path: Path, value: str) -> None:
    _atomic_write(path, value)


@dataclass(frozen=True)
class RunArtifacts:
    root: Path

    @property
    def raw_requests(self) -> Path: return self.root / "raw_requests"

    @property
    def raw_responses(self) -> Path: return self.root / "raw_responses"

    @property
    def parsed_arguments(self) -> Path: return self.root / "parsed_tool_arguments"

    def request(self, unit_id: str) -> Path: return self.raw_requests / f"{unit_id}_request.json"

    def response(self, unit_id: str) -> Path: return self.raw_responses / f"{unit_id}_response.json"

    def parsed(self, unit_id: str) -> Path: return self.parsed_arguments / f"{unit_id}.json"

    def path(self, name: str) -> Path: return self.root / name

    def json(self, name: str, value: Any) -> None: write_json(self.path(name), value)

    def text(self, name: str, value: str) -> None: write_text(self.path(name), value)
