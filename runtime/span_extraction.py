"""Deterministic aggregation for validated evidence-span extractions."""
from __future__ import annotations

from typing import Any

from prompts.contracts import FactExtraction


class ExtractionContractViolation(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def aggregate_namespaced_results(
    results: list[dict[str, Any]],
) -> list[tuple[str, FactExtraction]]:
    """Rebind namespaced technical IDs to the unchanged graph contract."""
    output: list[tuple[str, FactExtraction]] = []
    entity_cursor = 1
    for result in results:
        remap = {
            item["local_id"]: f"M{entity_cursor + index:03d}"
            for index, item in enumerate(result["entities"])
        }
        entity_cursor += len(remap)
        entities = []
        for item in result["entities"]:
            row = dict(item)
            row["local_id"] = remap[row["local_id"]]
            entities.append(row)
        relations = []
        for item in result["relations"]:
            row = dict(item)
            row.pop("relation_local_id")
            row["source_local_id"] = remap[row["source_local_id"]]
            row["target_local_id"] = remap[row["target_local_id"]]
            row["qualifiers"] = dict(row["qualifiers"])
            for key, candidate in row["qualifiers"].items():
                if isinstance(candidate, str) and candidate in remap:
                    row["qualifiers"][key] = remap[candidate]
            relations.append(row)
        output.append((result["unit_id"], FactExtraction.model_validate({
            "entities": entities,
            "relations": relations,
        })))
    return output
