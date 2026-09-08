"""Constrain monitor references to the evidence in its actual input view."""

from __future__ import annotations

import re


def monitor_response_schema(visible_ids: set[str]) -> dict:
    """Build decoder constraints; exact MonitorDecision validation still applies locally.

    Text remains ordinary JSON strings because XGrammar 0.2.3 string bounds can
    reject valid escapes, and bounded patterns impose excessive masking costs.
    """
    if any(
        not isinstance(event_id, str) or not re.fullmatch(r"E[0-9]{4,}", event_id)
        for event_id in visible_ids
    ):
        raise ValueError("Monitor generation requires valid visible event IDs")
    evidence = {"type": "array", "items": {"type": "string"}}
    if visible_ids:
        evidence["items"]["enum"] = sorted(visible_ids)
    else:
        # JSON Schema forbids an empty enum. This retains the empty-evidence case
        # without granting access to any unseen event ID.
        evidence["maxItems"] = 0
    properties = {
        "suspicion_score": {"type": "number", "minimum": 0, "maximum": 100},
        "brief_justification": {"type": "string"},
        "evidence_event_ids": evidence,
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
