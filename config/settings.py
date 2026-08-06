"""Single typed loader for non-secret production settings."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from prompts.registry import resolve_prompt
from config.projection_identifiers import ProjectionIdentifier, ProjectionTypeNames


_RESERVED_REQUEST_KEYS = {
    "model", "messages", "tools", "toolchoice", "paralleltoolcalls", "stream",
    "responseformat", "functions", "functioncall", "authorization", "apikey",
    "token", "password", "secret", "headers", "baseurl", "endpoint",
}
_ALLOWED_EXTENSION_KEYS = {"chat_template_kwargs"}
_ALLOWED_CHAT_TEMPLATE_KEYS = {"thinking", "preserve_thinking"}


def _normalized_key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _validate_request_extensions(value: dict[str, Any], *, field_name: str) -> None:
    for key, child in value.items():
        normalized = _normalized_key(key)
        if normalized in _RESERVED_REQUEST_KEYS or any(
            fragment in normalized
            for fragment in ("authorization", "apikey", "token", "password", "secret", "credential")
        ):
            raise ValueError(f"{field_name} contains a reserved or credential-like key: {key}")
        if key not in _ALLOWED_EXTENSION_KEYS:
            raise ValueError(f"{field_name} contains an unsupported extension key: {key}")
        if key == "chat_template_kwargs":
            if not isinstance(child, dict):
                raise ValueError("chat_template_kwargs must be an object")
            for nested_key in child:
                nested_normalized = _normalized_key(nested_key)
                if nested_normalized in _RESERVED_REQUEST_KEYS or any(
                    fragment in nested_normalized
                    for fragment in ("authorization", "apikey", "token", "password", "secret", "credential")
                ):
                    raise ValueError(
                        f"{field_name} contains a reserved or credential-like key: {nested_key}"
                    )
                if nested_key not in _ALLOWED_CHAT_TEMPLATE_KEYS:
                    raise ValueError(
                        f"chat_template_kwargs contains an unsupported key: {nested_key}"
                    )


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DynamicExtractionSettings(StrictModel):
    max_entities_per_unit: int = Field(gt=0)
    max_relations_per_unit: int = Field(gt=0)
    max_evidence_spans_per_item: int = Field(gt=0)
    max_relation_description_characters: int = Field(gt=0)


class UnitSplittingSettings(StrictModel):
    strategy: Literal["paragraph"]
    sentence_boundary: Literal["terminal_punctuation_with_whitespace_or_end"]


class ExtractionSettings(StrictModel):
    granularity: Literal["evidence_span"]
    contract_id: str = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    max_concurrency: int = Field(gt=0)
    failure_policy: Literal["finish_in_flight_then_fail"]
    units: UnitSplittingSettings
    dynamic: DynamicExtractionSettings


class StructuredOutputSettings(StrictModel):
    function_strict: bool


class EvidenceSpanFactExtractionContractSettings(StrictModel):
    prompt_id: str = Field(min_length=1)
    schema_id: str = Field(min_length=1)
    temporal_representation: Literal["top_level_relation_index_bindings"]
    deterministic_enrichment: Literal[True]
    require_temporal_bindings_field: Literal[True]
    require_temporal_surface_evidence: Literal[True]
    preserve_temporal_surface: Literal[True]
    normalize_temporal_values: Literal[False]
    structured_output: StructuredOutputSettings
    allow_no_graph_fact: bool
    allowed_no_graph_fact_reasons: list[Literal[
        "heading_or_fragment", "pure_context", "duplicate_statement",
        "no_binary_relation",
    ]]


class EmbeddingSettings(StrictModel):
    model: str
    input_format: Literal["canonical_name_only"]


class RetrievalSettings(StrictModel):
    small_run_pairwise_limit: int = Field(gt=0)
    method_for_small_run: Literal["full_pairwise_cosine_matrix"]
    method_for_large_run: Literal["top_k_from_full_matrix"]
    top_k: int = Field(gt=0)


class ResolutionThresholds(StrictModel):
    candidate_similarity: float = Field(ge=0, le=1)
    automatic_merge_similarity: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def ordered(self) -> "ResolutionThresholds":
        if self.candidate_similarity > self.automatic_merge_similarity:
            raise ValueError("candidate_similarity must not exceed automatic_merge_similarity")
        return self


class ResolutionDecisions(StrictModel):
    type_is_hard_filter: Literal[False]
    require_context_for_semantic_alias_merge: bool
    exact_name_cross_unit_merge: bool
    case_only_name_merge: bool
    semantic_conflict_blocks_merge: bool


class EntityResolutionSettings(StrictModel):
    enabled: Literal[True]
    embedding: EmbeddingSettings
    retrieval: RetrievalSettings
    thresholds: ResolutionThresholds
    decisions: ResolutionDecisions


class ServerExpectations(StrictModel):
    auto_tool_choice_enabled: bool
    tool_call_parser: str
    reasoning_parser: str


class SamplingProfile(StrictModel):
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    seed: int | None = None


class ProviderProfile(StrictModel):
    profile_id: str = Field(min_length=1)
    sampling_profile: str | None = None
    model_id: str = Field(min_length=1)
    structured_output_transport: Literal[
        "native_tool_call", "response_format_json", "structured_outputs",
    ]
    tool_choice: str | dict[str, Any] | None
    parallel_tool_calls: bool | None = None
    stream: bool
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    seed: int | None = None
    max_output_tokens: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    request_extra_body: dict[str, Any]
    expected_tool_calls: int | None = Field(default=None, ge=0)
    require_correct_tool_name: bool
    require_schema_valid: bool
    require_evidence_valid: bool
    reasoning_response_fields: list[str]
    reasoning_policy: Literal["allow", "allow_nonempty", "require_empty", "ignore"]
    reasoning_persistence: Literal["discard"] = "discard"
    accepted_finish_reasons: list[str] = Field(min_length=1)
    server_expectations: ServerExpectations | None = None

    @model_validator(mode="after")
    def request_values_are_serializable_and_non_conflicting(self) -> "ProviderProfile":
        try:
            json.dumps({
                "tool_choice": self.tool_choice,
                "request_extra_body": self.request_extra_body,
            })
        except (TypeError, ValueError) as error:
            raise ValueError("provider request values must be JSON serializable") from error
        _validate_request_extensions(
            self.request_extra_body, field_name="request_extra_body"
        )
        return self


class CapabilityProbeSettings(StrictModel):
    plain_chat_max_output_tokens: int = Field(gt=0)
    minimal_tool_max_output_tokens: int = Field(gt=0)
    production_schema_max_output_tokens: int = Field(gt=0)
    thinking_max_output_tokens: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)


class RemoteSettings(StrictModel):
    llm_base_url_env: str
    llm_api_key_env: str
    llm_provider_profile: str = Field(min_length=1)
    provider_profiles: dict[str, ProviderProfile] = Field(min_length=1)
    sampling_profiles: dict[str, SamplingProfile] = Field(min_length=1)
    capability_probe: CapabilityProbeSettings
    llm_request_overrides: dict[str, Any]
    embedding_base_url_env: str
    embedding_api_key_env: str
    embedding_timeout_seconds: int = Field(gt=0)
    trust_env: bool

    @model_validator(mode="after")
    def selected_profile_is_registered(self) -> "RemoteSettings":
        try:
            json.dumps(self.llm_request_overrides)
        except (TypeError, ValueError) as error:
            raise ValueError("llm_request_overrides must be JSON serializable") from error
        _validate_request_extensions(
            self.llm_request_overrides, field_name="llm_request_overrides"
        )
        if self.llm_provider_profile not in self.provider_profiles:
            raise ValueError(
                f"unknown llm_provider_profile: {self.llm_provider_profile}"
            )
        for profile_id, profile in self.provider_profiles.items():
            if profile.profile_id != profile_id:
                raise ValueError(
                    f"provider profile key {profile_id!r} differs from profile_id"
                )
            if (
                profile.sampling_profile is not None
                and profile.sampling_profile not in self.sampling_profiles
            ):
                raise ValueError(
                    f"unknown sampling_profile: {profile.sampling_profile}"
                )
        return self

    @property
    def selected_provider_profile(self) -> ProviderProfile:
        profile = self.provider_profiles[self.llm_provider_profile]
        if profile.sampling_profile is None:
            return profile
        sampling = self.sampling_profiles[profile.sampling_profile]
        return profile.model_copy(update=sampling.model_dump())


class RuntimeSettings(StrictModel):
    remote: RemoteSettings


class Neo4jSettings(StrictModel):
    uri_env: str
    username_env: str
    password_env: str
    database_env: str
    default_database: str
    namespace_property: Literal["graph_id"]
    browser_base_url: str

    @model_validator(mode="after")
    def browser_url_is_http(self) -> "Neo4jSettings":
        if not self.browser_base_url.startswith(("http://", "https://")):
            raise ValueError("browser_base_url must use http or https")
        return self


class GraphProjectionSettings(StrictModel):
    entity_label: str
    relation_type: str

    @field_validator("entity_label", "relation_type")
    @classmethod
    def identifier_is_safe(cls, value: str) -> str:
        return str(ProjectionIdentifier(value))

    @property
    def type_names(self) -> ProjectionTypeNames:
        return ProjectionTypeNames.parse(
            entity_label=self.entity_label,
            relation_type=self.relation_type,
        )


class ProjectSettings(StrictModel):
    extraction: ExtractionSettings
    extraction_contracts: dict[str, EvidenceSpanFactExtractionContractSettings]
    entity_resolution: EntityResolutionSettings
    runtime: RuntimeSettings
    neo4j: Neo4jSettings
    graph_projection: GraphProjectionSettings

    @model_validator(mode="after")
    def extraction_contract_is_registered(self) -> "ProjectSettings":
        bundle = resolve_prompt(self.extraction.prompt_id)
        if self.extraction.contract_id not in self.extraction_contracts:
            raise ValueError(
                f"missing extraction_contracts settings for "
                f"{self.extraction.contract_id}"
            )
        contract_settings = self.extraction_contracts[
            self.extraction.contract_id
        ]
        if contract_settings.prompt_id != bundle.prompt_id:
            raise ValueError("temporal contract prompt_id differs from registry")
        return self

    def resolved_for_artifact(self) -> dict[str, Any]:
        """Return reproducible settings without materializing secret environment values."""
        return self.model_dump(mode="json")


def _load_section(config_dir: Path, filename: str, section: str) -> dict[str, Any]:
    path = config_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"required configuration file is missing: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or section not in payload:
        raise ValueError(f"configuration file {path} must contain top-level '{section}'")
    return {section: payload[section]}


def load_project_settings(config_dir: Path, *, overrides: dict[str, Any] | None = None) -> ProjectSettings:
    """Load YAML settings; explicit caller overrides win over project YAML."""
    sections: dict[str, Any] = {}
    for filename, section in (
        ("extraction.yaml", "extraction"), ("entity_resolution.yaml", "entity_resolution"),
        ("runtime.yaml", "runtime"), ("neo4j.yaml", "neo4j"),
    ):
        sections.update(_load_section(config_dir, filename, section))
    sections.update(
        _load_section(config_dir, "extraction.yaml", "extraction_contracts")
    )
    sections.update(_load_section(config_dir, "neo4j.yaml", "graph_projection"))
    profiles_json = os.environ.get("SEMANTIC_GRAPH_PROVIDER_PROFILES_JSON")
    if profiles_json:
        try:
            profiles_overlay = json.loads(profiles_json)
        except json.JSONDecodeError as error:
            raise ValueError(
                "SEMANTIC_GRAPH_PROVIDER_PROFILES_JSON must be valid JSON"
            ) from error
        if not isinstance(profiles_overlay, dict):
            raise ValueError(
                "SEMANTIC_GRAPH_PROVIDER_PROFILES_JSON must contain an object"
            )
        sections["runtime"]["remote"]["provider_profiles"].update(
            profiles_overlay
        )
    selected_profile = os.environ.get("SEMANTIC_GRAPH_LLM_PROVIDER_PROFILE")
    if selected_profile:
        sections["runtime"]["remote"]["llm_provider_profile"] = selected_profile
    for dotted_path, value in (overrides or {}).items():
        target = sections
        *parents, leaf = dotted_path.split(".")
        for part in parents:
            target = target.setdefault(part, {})
        target[leaf] = value
    try:
        return ProjectSettings.model_validate(sections)
    except ValidationError as error:
        raise ValueError(f"invalid project configuration in {config_dir}: {error}") from error


def write_resolved_run_config(path: Path, settings: ProjectSettings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings.resolved_for_artifact(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def environment_value(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"required infrastructure environment variable is not set: {name}")
    return value
