"""Persistent accounting for primary HTTP attempts and forbidden semantic calls."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class PostExtractionModelCallError(RuntimeError):
    """Raised before a production post-extraction model request can run."""


@dataclass
class UnitModelCallBudget:
    unit_id: str
    primary_successful_calls: int = 0
    primary_http_attempts: int = 0
    primary_failed_attempts: int = 0
    primary_timeout_attempts: int = 0
    transport_successes: int = 0
    transport_failures: int = 0
    provider_response_rejections: int = 0
    tool_contract_failures: int = 0
    schema_failures: int = 0
    evidence_failures: int = 0
    accepted_extractions: int = 0
    terminal_outcome: str | None = None
    terminal_subcategory: str | None = None
    automatic_retries: int = 0
    post_extraction_llm_calls: int = 0
    embedding_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    thinking_responses: int = 0
    schema_valid_extractions: int = 0
    evidence_valid_extractions: int = 0
    persisted_span_units: int = 0

    def record_http_attempt(self, *, automatic_retry: bool = False) -> None:
        self.primary_http_attempts += 1
        if automatic_retry: self.automatic_retries += 1

    def _terminal(self, outcome: str, subcategory: str) -> None:
        if self.terminal_outcome is not None:
            raise RuntimeError(f"unit {self.unit_id} already has a terminal outcome")
        self.terminal_outcome = outcome
        self.terminal_subcategory = subcategory

    def record_transport_failure(self, *, timed_out: bool = False, subcategory: str) -> None:
        self.primary_failed_attempts += 1
        self.transport_failures += 1
        if timed_out: self.primary_timeout_attempts += 1
        self._terminal("transport_failed", subcategory)

    def record_transport_success(
        self, *, prompt_tokens: int, completion_tokens: int,
        reasoning_present: bool = False,
    ) -> None:
        self.transport_successes += 1
        self.prompt_tokens += prompt_tokens; self.completion_tokens += completion_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        if reasoning_present:
            self.thinking_responses += 1

    def record_rejection(self, outcome: str, subcategory: str) -> None:
        field_by_outcome = {
            "provider_response_rejected": "provider_response_rejections",
            "tool_contract_failed": "tool_contract_failures",
            "schema_failed": "schema_failures",
            "evidence_failed": "evidence_failures",
        }
        field = field_by_outcome[outcome]
        setattr(self, field, getattr(self, field) + 1)
        self._terminal(outcome, subcategory)

    def record_accepted(self) -> None:
        self.primary_successful_calls += 1
        self.accepted_extractions += 1
        self._terminal("accepted", "accepted")

    def record_embedding(self, calls: int = 1) -> None:
        self.embedding_calls += calls

    def record_schema_valid(self) -> None:
        self.schema_valid_extractions += 1

    def record_evidence_valid(self) -> None:
        self.evidence_valid_extractions += 1

    def record_persisted_span(self) -> None:
        self.persisted_span_units += 1

    def reject_post_extraction_llm_call(self, stage: str) -> None:
        self.post_extraction_llm_calls += 1
        raise PostExtractionModelCallError(f"post-extraction LLM stage '{stage}' is forbidden in the single-pass production pipeline")

    def assert_valid(self) -> None:
        if self.primary_successful_calls != 1:
            raise RuntimeError(f"unit {self.unit_id} requires exactly one successful primary extraction call")
        terminal_rejections = (
            self.provider_response_rejections + self.tool_contract_failures
            + self.schema_failures + self.evidence_failures
        )
        if self.primary_successful_calls != self.accepted_extractions:
            raise RuntimeError(f"unit {self.unit_id} accepted extraction accounting is inconsistent")
        if self.primary_failed_attempts != self.transport_failures:
            raise RuntimeError(f"unit {self.unit_id} transport failure accounting is inconsistent")
        if self.primary_http_attempts != self.accepted_extractions + self.transport_failures + terminal_rejections:
            raise RuntimeError(f"unit {self.unit_id} HTTP attempt accounting is inconsistent")
        if self.transport_successes + self.transport_failures != self.primary_http_attempts:
            raise RuntimeError(f"unit {self.unit_id} transport accounting is inconsistent")
        if self.terminal_outcome is None:
            raise RuntimeError(f"unit {self.unit_id} requires exactly one terminal outcome")
        if self.accepted_extractions != 1 or self.terminal_outcome != "accepted":
            raise RuntimeError(f"unit {self.unit_id} requires exactly one accepted extraction")
        if self.post_extraction_llm_calls != 0:
            raise PostExtractionModelCallError(f"unit {self.unit_id} recorded a forbidden post-extraction LLM call")
        if self.schema_valid_extractions != 1:
            raise RuntimeError(f"unit {self.unit_id} requires exactly one schema-valid extraction")
        if self.evidence_valid_extractions != 1:
            raise RuntimeError(f"unit {self.unit_id} requires exactly one evidence-valid extraction")
        if self.persisted_span_units != 1:
            raise RuntimeError(f"unit {self.unit_id} requires exactly one persisted extraction unit")

    def as_dict(self) -> dict[str, int | str]: return asdict(self)


class ProductionModelCallBudget:
    def __init__(self) -> None: self._units: dict[str, UnitModelCallBudget] = {}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProductionModelCallBudget":
        budget = cls()
        for row in payload.get("units", []):
            fields = UnitModelCallBudget.__dataclass_fields__
            budget._units[row["unit_id"]] = UnitModelCallBudget(**{key:value for key,value in row.items() if key in fields})
        return budget

    def unit(self, unit_id: str) -> UnitModelCallBudget:
        return self._units.setdefault(unit_id, UnitModelCallBudget(unit_id=unit_id))

    def assert_valid(self) -> None:
        for item in self._units.values(): item.assert_valid()

    def as_dict(self) -> dict[str, object]:
        units = [
            self._units[unit_id].as_dict()
            for unit_id in sorted(self._units)
        ]
        total_fields = ("primary_successful_calls","primary_http_attempts","primary_failed_attempts","primary_timeout_attempts",
                        "transport_successes","transport_failures","provider_response_rejections","tool_contract_failures","schema_failures","evidence_failures","accepted_extractions","automatic_retries","post_extraction_llm_calls","embedding_calls",
                        "prompt_tokens","completion_tokens","total_tokens","thinking_responses",
                        "schema_valid_extractions","evidence_valid_extractions","persisted_span_units")
        return {"units":units,"totals":{field:sum(int(item[field]) for item in units) for field in total_fields}}
