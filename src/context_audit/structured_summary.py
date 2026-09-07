"""Constrained summary drafts and a deterministic index of existing citations."""

from __future__ import annotations

import json
import re

from context_audit.representations import evidence_ids

CLAIM_FIELDS = (
    "environment_and_state",
    "actions_and_observed_results",
    "important_identifiers",
    "contradictions_or_missing_information",
)


def structured_summary_schema(visible_ids: set[str]) -> dict:
    """Return an independent schema; its claims, never an index, are model-generated."""
    if not visible_ids or any(
        not isinstance(event_id, str) or not re.fullmatch(r"E[0-9]{4,}", event_id)
        for event_id in visible_ids
    ):
        raise ValueError("Structured generation requires visible event IDs")
    return {
        "type": "object",
        "properties": {
            field: {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "evidence_event_ids": {
                            "type": "array", "minItems": 1,
                            "items": {"type": "string", "enum": sorted(visible_ids)},
                        },
                    },
                    "required": ["text", "evidence_event_ids"],
                    "additionalProperties": False,
                },
            }
            for field in CLAIM_FIELDS
        },
        "required": list(CLAIM_FIELDS),
        "additionalProperties": False,
    }


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate structured draft keys")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ValueError("Structured draft is not strict JSON")


def assemble_structured_summary(raw_text: str, visible_ids: set[str]) -> str:
    """Append only model-selected citations; preserve claim text and never infer evidence."""
    try:
        data = json.loads(
            raw_text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except json.JSONDecodeError as exc:
        raise ValueError("Summary is not valid JSON") from exc
    if not isinstance(data, dict) or set(data) != set(CLAIM_FIELDS):
        raise ValueError("Structured draft fields do not match schema")
    cited, assembled = set(), {}
    for field in CLAIM_FIELDS:
        claims = data[field]
        if not isinstance(claims, list):
            raise ValueError("Structured draft claims must be arrays of citation objects")
        assembled[field] = []
        for claim in claims:
            if not isinstance(claim, dict) or set(claim) != {"text", "evidence_event_ids"}:
                raise ValueError("Structured draft claim fields do not match schema")
            text, selected = claim["text"], claim["evidence_event_ids"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Structured draft claims require nonempty text")
            if not isinstance(selected, list) or not all(
                isinstance(item, str) for item in selected
            ):
                raise ValueError("Invalid source_event_ids")
            if not selected:
                raise ValueError("Each structured claim requires a visible event citation")
            selected = set(selected)
            mentioned = evidence_ids(text)
            if not (selected | mentioned) <= visible_ids:
                raise ValueError("Missing or unknown source event IDs")
            if not mentioned <= selected:
                raise ValueError("Claim text references event IDs not selected for this item")
            cited.update(selected)
            assembled[field].append(text + " " + " ".join(f"[{item}]" for item in sorted(selected)))
    if visible_ids and not cited:
        raise ValueError("Missing or unknown source event IDs")
    assembled["source_event_ids"] = sorted(cited)
    return json.dumps(assembled, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
