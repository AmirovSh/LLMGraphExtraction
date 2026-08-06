"""OpenAI-compatible embedding transport and observable metadata artifacts."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

import httpx


def _vector_hash(vector: list[float]) -> str:
    encoded = json.dumps(vector, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def request_embeddings(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    inputs: list[dict[str, Any]],
    timeout_seconds: int,
    trust_env: bool,
    write_artifact: Callable[[str, Any], None],
    post: Callable[..., Any] = httpx.post,
) -> tuple[list[list[float]], dict[str, Any], int]:
    request = {"model": model, "input": [item["embedding_text"] for item in inputs]}
    write_artifact("entity_embedding_inputs.json", {"inputs": inputs, "input_count": len(inputs)})
    write_artifact("embedding_request.json", request)
    response = post(
        f"{endpoint.rstrip('/')}/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json=request,
        timeout=timeout_seconds,
        trust_env=trust_env,
    )
    response.raise_for_status()
    body = response.json()
    data = body.get("data")
    if not isinstance(data, list):
        raise ValueError("embedding response data must be a list")
    if len(data) != len(inputs):
        raise ValueError(
            f"embedding response vector count {len(data)} does not match input count {len(inputs)}"
        )
    expected_indexes = list(range(len(inputs)))
    indexes = [item.get("index") for item in data if isinstance(item, dict)]
    if len(indexes) != len(data) or any(not isinstance(index, int) for index in indexes):
        raise ValueError("embedding response indexes must be integer input positions")
    indexes.sort()
    if indexes != expected_indexes:
        raise ValueError("embedding response indexes must match input positions exactly")
    data = sorted(data, key=lambda item: item["index"])
    if any(not isinstance(item.get("embedding"), list) for item in data):
        raise ValueError("embedding response vectors must be lists")
    vectors = [item["embedding"] for item in data]
    dimension = len(vectors[0]) if vectors else 0
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("embedding response vectors have inconsistent dimensions")
    write_artifact(
        "embedding_response_metadata.json",
        {
            "http_status": response.status_code,
            "model": body.get("model", model),
            "usage": body.get("usage") or {},
            "data_count": len(data),
            "vector_dimension": dimension,
            "vector_hashes": [
                {"index": item["index"], "hash_sha256": _vector_hash(item["embedding"])} for item in data
            ],
            "vectors_saved": False,
        },
    )
    return vectors, body, dimension
