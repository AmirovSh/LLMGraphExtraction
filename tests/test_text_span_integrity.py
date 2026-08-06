from __future__ import annotations

import pytest

from prompts.registry import resolve_prompt
from runtime.production_inputs import (
    render_prompt, sentence_spans, split_unit_spans, split_units,
)


@pytest.mark.parametrize(
    "source",
    [
        "Version 3.6 is active.",
        "IP address 10.20.30.40 is configured.",
        "Threshold is 0.75.",
        "Release occurred on 2026-07-30.",
        "Value changed from 2.5 to 3.0.",
    ],
)
def test_structured_tokens_are_not_split(source: str) -> None:
    spans = sentence_spans(source)
    assert [span["text"] for span in spans] == [source]


def test_sentence_span_offsets_round_trip_exactly_without_strip_assertion() -> None:
    source = (
        "  Version 3.6 — “alpha”  is active.  \n"
        "Unicode Журнал keeps  multiple   spaces!  "
    )
    spans = sentence_spans(source, document_start_offset=11)
    for span in spans:
        assert source[span["start_offset"]:span["end_offset"]] == span["text"]
        assert span["document_start_offset"] == 11 + span["start_offset"]
        assert span["document_end_offset"] == 11 + span["end_offset"]
    assert spans[0]["text"] == "Version 3.6 — “alpha”  is active."
    assert spans[1]["text"] == "Unicode Журнал keeps  multiple   spaces!"


def test_document_unit_and_sentence_offsets_round_trip() -> None:
    source = (
        " \tVersion 3.6 is active.\n"
        "Second line uses  0.75.  \n\n"
        "  IP address 10.20.30.40 is configured. \n"
    )
    units = split_unit_spans(source, strategy="paragraph")
    assert split_units(source, strategy="paragraph") == [unit.text for unit in units]
    for unit in units:
        assert source[unit.start_offset:unit.end_offset] == unit.text
        spans = sentence_spans(
            unit.text, document_start_offset=unit.start_offset,
        )
        for span in spans:
            assert unit.text[span["start_offset"]:span["end_offset"]] == span["text"]
            assert (
                source[span["document_start_offset"]:span["document_end_offset"]]
                == span["text"]
            )


def test_prompt_payload_keeps_offset_metadata_out_of_prompt_text() -> None:
    spans = sentence_spans("Version 3.6 is active.")
    rendered = render_prompt(
        resolve_prompt("prompt_kimi_default"), spans[0]["text"], spans,
        unit_id="U001_S001",
    )
    assert '"span_id":"S001"' in rendered
    assert '"text":"Version 3.6 is active."' in rendered
    assert "start_offset" not in rendered
    assert "document_start_offset" not in rendered
