"""Render observable data, without inferring human permission from tool roles."""

from __future__ import annotations

import json

from context_audit.schemas import TranscriptInput


def render_task_header(transcript: TranscriptInput) -> str:
    """The initial human text is preserved literally, outside the body budget."""
    environment = (
        "<initial_cwd>" + json.dumps(transcript.initial_cwd) + "</initial_cwd>\n"
        if transcript.initial_cwd is not None
        else ""
    )
    return environment + "<initial_task>\n" + transcript.initial_task + "\n</initial_task>\n"


def render_body(transcript: TranscriptInput) -> str:
    """JSON lines keep roles and event kinds explicit; no event is executed."""
    rows = []
    for event in transcript.events:
        fields = event.model_dump(exclude_none=True)
        rows.append(json.dumps(fields, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(rows)
