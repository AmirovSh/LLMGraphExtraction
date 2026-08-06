"""Parse the two supported structured-output transports without semantic repair."""
from __future__ import annotations

import json
from typing import Any

def parse_tool_output(
    message: dict[str, Any],
    *,
    transport: str = "native_tool_call",
    expected_function: str = "submit_dynamic_inventory",
) -> tuple[dict[str, Any], str]:
    if transport != "native_tool_call":
        raise RuntimeError(f"unsupported structured-output transport: {transport}")
    calls = message.get("tool_calls") or []
    if len(calls) != 1:
        raise RuntimeError("native_tool_call requires exactly one tool call")
    function = calls[0].get("function") or {}
    if function.get("name") != expected_function:
        raise RuntimeError("native tool call used the wrong function name")
    payload = json.loads(function.get("arguments", ""))
    if not isinstance(payload, dict):
        raise RuntimeError("native tool-call arguments must be a JSON object")
    return payload, "native_tool_call"
