"""Strict, side-effect-free multi-document batch manifest resolution."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from config.settings import ProjectSettings
from prompts.registry import resolve_prompt
from runtime.production_inputs import effective_request_identity


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BatchHeader(StrictModel):
    id: str = Field(min_length=1)
    output_directory: Path


class BatchExecution(StrictModel):
    global_max_in_flight: int = Field(gt=0)
    per_document_max_in_flight: int = Field(gt=0)
    per_graph_max_in_flight: int = Field(gt=0)
    scheduling_policy: Literal["fair_round_robin"] = "fair_round_robin"
    continue_other_graphs_on_failure: bool = True

    @model_validator(mode="after")
    def limits_are_coherent(self) -> "BatchExecution":
        if self.per_document_max_in_flight > self.per_graph_max_in_flight:
            raise ValueError("per_document_max_in_flight must not exceed per_graph_max_in_flight")
        if self.per_graph_max_in_flight > self.global_max_in_flight:
            raise ValueError("per_graph_max_in_flight must not exceed global_max_in_flight")
        return self


class BatchModel(StrictModel):
    provider_profile: str = Field(min_length=1)
    model_id: str = Field(min_length=1)


class BatchDocument(StrictModel):
    id: str = Field(min_length=1)
    path: Path


class BatchGraph(StrictModel):
    id: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1)


class BatchManifest(StrictModel):
    manifest_version: Literal[1]
    batch: BatchHeader
    execution: BatchExecution
    model: BatchModel
    documents: list[BatchDocument] = Field(min_length=1)
    graphs: list[BatchGraph] = Field(min_length=1)

    @model_validator(mode="after")
    def identifiers_and_membership(self) -> "BatchManifest":
        for kind, values in (
            ("batch", [self.batch.id]),
            ("document", [item.id for item in self.documents]),
            ("graph", [item.id for item in self.graphs]),
        ):
            invalid = [value for value in values if not _ID.fullmatch(value)]
            if invalid:
                raise ValueError(f"invalid {kind} ID: {invalid[0]}")
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {kind} ID")
        declared = {item.id for item in self.documents}
        referenced = [document for graph in self.graphs for document in graph.documents]
        unknown = sorted(set(referenced) - declared)
        if unknown:
            raise ValueError(f"unknown document reference: {unknown}")
        duplicates = sorted({item for item in referenced if referenced.count(item) > 1})
        if duplicates:
            raise ValueError(f"documents belong to multiple graphs: {duplicates}")
        missing = sorted(declared - set(referenced))
        if missing:
            raise ValueError(f"documents belong to no graph: {missing}")
        return self


@dataclass(frozen=True)
class ResolvedDocumentPlan:
    document_id: str
    path: Path
    source_filename: str
    source_sha256: str
    source_size_bytes: int
    graph_id: str


@dataclass(frozen=True)
class ResolvedGraphPlan:
    graph_id: str
    document_ids: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedBatchPlan:
    manifest: BatchManifest
    manifest_path: Path
    output_directory: Path
    documents: tuple[ResolvedDocumentPlan, ...]
    graphs: tuple[ResolvedGraphPlan, ...]
    request_identity: dict[str, object]

    def public_dict(self) -> dict[str, object]:
        return {
            "manifest_version": self.manifest.manifest_version,
            "batch_id": self.manifest.batch.id,
            "output_directory": self.output_directory.name,
            "execution": self.manifest.execution.model_dump(mode="json"),
            "model": self.manifest.model.model_dump(mode="json"),
            "documents": [
                {
                    "document_id": item.document_id,
                    "source_filename": item.source_filename,
                    "source_sha256": item.source_sha256,
                    "source_size_bytes": item.source_size_bytes,
                    "graph_id": item.graph_id,
                }
                for item in self.documents
            ],
            "graphs": [
                {"graph_id": item.graph_id, "document_ids": list(item.document_ids)}
                for item in self.graphs
            ],
            "request_identity": self.request_identity,
        }


def load_batch_manifest(path: Path) -> BatchManifest:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return BatchManifest.model_validate(payload)


def resolve_batch_plan(
    manifest_path: Path, settings: ProjectSettings, *, output_override: Path | None = None,
) -> ResolvedBatchPlan:
    manifest_path = manifest_path.resolve()
    manifest = load_batch_manifest(manifest_path)
    profile = settings.runtime.remote.provider_profiles.get(manifest.model.provider_profile)
    if profile is None:
        raise ValueError(f"unknown provider profile: {manifest.model.provider_profile}")
    if profile.model_id != manifest.model.model_id:
        raise ValueError("manifest model_id differs from the selected provider profile")
    output = (output_override or manifest.batch.output_directory)
    if not output.is_absolute():
        output = (manifest_path.parent / output).resolve()
    if output.exists():
        raise FileExistsError("batch output directory already exists")
    membership = {
        document_id: graph.id for graph in manifest.graphs for document_id in graph.documents
    }
    documents: list[ResolvedDocumentPlan] = []
    for item in manifest.documents:
        source_path = item.path if item.path.is_absolute() else manifest_path.parent / item.path
        source_path = source_path.resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"document is missing or unreadable: {item.id}")
        data = source_path.read_bytes()
        documents.append(ResolvedDocumentPlan(
            document_id=item.id, path=source_path, source_filename=source_path.name,
            source_sha256=hashlib.sha256(data).hexdigest(), source_size_bytes=len(data),
            graph_id=membership[item.id],
        ))
    bundle = resolve_prompt(settings.extraction.prompt_id)
    dynamic = settings.extraction.dynamic
    schema = bundle.schema_builder(
        max_entities=dynamic.max_entities_per_unit,
        max_relations=dynamic.max_relations_per_unit,
        max_evidence_spans=dynamic.max_evidence_spans_per_item,
        max_relation_description_characters=dynamic.max_relation_description_characters,
    )
    selected_settings = settings.model_copy(deep=True)
    selected_settings.runtime.remote.llm_provider_profile = manifest.model.provider_profile
    identity = effective_request_identity(selected_settings, bundle, schema)
    return ResolvedBatchPlan(
        manifest=manifest, manifest_path=manifest_path, output_directory=output,
        documents=tuple(documents),
        graphs=tuple(ResolvedGraphPlan(graph_id=g.id, document_ids=tuple(g.documents)) for g in manifest.graphs),
        request_identity=json.loads(json.dumps(identity)),
    )
