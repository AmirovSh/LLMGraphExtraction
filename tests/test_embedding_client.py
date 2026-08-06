from __future__ import annotations

import pytest

from runtime.embedding_client import request_embeddings


class Response:
    status_code = 200

    def __init__(self, data=None):
        self.data = data or [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]},
        ]

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "model": "bge-m3",
            "usage": {"prompt_tokens": 2},
            "data": self.data,
        }


def test_embedding_transport_preserves_request_order_and_observability() -> None:
    artifacts = {}
    captured = {}
    inputs = [
        {"local_entity_id": "M001", "embedding_text": "Build Registry"},
        {"local_entity_id": "M002", "embedding_text": "Record"},
    ]

    def post(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return Response()

    vectors, body, dimension = request_embeddings(
        endpoint="https://embed.invalid/v1",
        api_key="secret",
        model="bge-m3",
        inputs=inputs,
        timeout_seconds=120,
        trust_env=False,
        write_artifact=artifacts.__setitem__,
        post=post,
    )
    assert captured["args"][0] == "https://embed.invalid/v1/embeddings"
    assert captured["kwargs"]["json"] == {"model": "bge-m3", "input": ["Build Registry", "Record"]}
    assert vectors == [[1.0, 0.0], [0.0, 1.0]] and dimension == 2 and body["model"] == "bge-m3"
    assert artifacts["entity_embedding_inputs.json"]["inputs"] == inputs
    assert artifacts["embedding_response_metadata.json"]["vectors_saved"] is False
    assert len(artifacts["embedding_response_metadata.json"]["vector_hashes"]) == 2

    invalid_responses = [
        (
            Response([{"index": 0, "embedding": [1.0, 0.0]}]),
            "vector count 1 does not match input count 2",
        ),
        (
            Response(
                [
                    {"index": 0, "embedding": [1.0, 0.0]},
                    {"index": 2, "embedding": [0.0, 1.0]},
                ]
            ),
            "indexes must match input positions exactly",
        ),
        (
            Response(
                [
                    {"index": 0, "embedding": [1.0, 0.0]},
                    {"index": 1, "embedding": [0.0]},
                ]
            ),
            "inconsistent dimensions",
        ),
        (
            Response(
                [
                    {"index": 0, "embedding": [1.0, 0.0]},
                    {"index": 1},
                ]
            ),
            "vectors must be lists",
        ),
    ]
    for invalid_response, error_match in invalid_responses:
        with pytest.raises(ValueError, match=error_match):
            request_embeddings(
                endpoint="https://embed.invalid/v1",
                api_key="secret",
                model="bge-m3",
                inputs=inputs,
                timeout_seconds=120,
                trust_env=False,
                write_artifact=lambda *_: None,
                post=lambda *_, response=invalid_response, **__: response,
            )
