import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from context_audit.dataset import parse_transcript
from context_audit.render import render_body, render_task_header
from context_audit.schemas import Event, TranscriptInput

FIXTURE = Path(__file__).parent / "fixtures" / "visible_events.jsonl"


@pytest.mark.parametrize(
    "field",
    ["label", "category", "metadata", "description", "outcome", "mechanism", "scores", "summary"],
)
def test_model_input_types_reject_evaluator_fields(field):
    with pytest.raises(ValidationError):
        TranscriptInput(
            transcript_id="t_" + "a" * 24,
            initial_task="Count blocks",
            events=[],
            **{field: "secret evaluator annotation"},
        )
    with pytest.raises(ValidationError):
        Event(
            event_id="E0001",
            role="assistant",
            kind="text",
            cwd="/demo",
            text="Visible",
            **{field: "secret evaluator annotation"},
        )


def test_sidecar_labels_and_directory_names_cannot_change_model_text(tmp_path):
    rendered = []
    for dirname, label in [("independent_a", 0), ("unrelated_b", 1)]:
        directory = tmp_path / dirname
        directory.mkdir()
        path = directory / "transcript.jsonl"
        path.write_bytes(FIXTURE.read_bytes())
        (directory / "metadata.json").write_text(
            json.dumps({"label": label, "scenario": "SECRET_SCENARIO", "summary": "SECRET_SUMMARY"})
        )
        t = parse_transcript(path, "t_" + "f" * 24, data_origin="synthetic_fixture")
        rendered.append(render_task_header(t) + render_body(t))
    assert rendered[0] == rendered[1]
    assert "SECRET_" not in rendered[0]
