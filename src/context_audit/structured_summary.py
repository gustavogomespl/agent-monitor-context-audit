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


def draft_limits(token_budget: int) -> dict[str, int]:
    """Structural limits, not a character-to-token estimate or final token guarantee."""
    if type(token_budget) is not int or token_budget < 4:
        raise ValueError("Bounded drafts require a token budget of at least four")
    return {"claims_per_field": 2, "references_per_claim": 2, "text_units": token_budget // 4}


def bounded_text_pattern(units: int) -> str:
    # XGrammar 0.2.3 applies string patterns to JSON's serialized string body.
    # An explicit JSON unit permits escaped quotes/backslashes and Unicode while
    # excluding unescaped delimiters/control bytes. maxLength in that compiler
    # rejects legitimate escapes. A surrogate-pair escape uses two units here.
    return rf'^([^"\\\x00-\x1f]|\\(["\\/bfnrt]|u[0-9a-fA-F]{{4}})){{0,{units}}}$'


def structured_summary_schema(visible_ids: set[str], *, token_budget: int | None = None) -> dict:
    """Return an independent schema; its claims, never an index, are model-generated."""
    if not visible_ids or any(
        not isinstance(event_id, str) or not re.fullmatch(r"E[0-9]{4,}", event_id)
        for event_id in visible_ids
    ):
        raise ValueError("Structured generation requires visible event IDs")
    schema = {
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
    if token_budget is not None:
        limits = draft_limits(token_budget)
        for field in CLAIM_FIELDS:
            array = schema["properties"][field]
            array["maxItems"] = limits["claims_per_field"]
            properties = array["items"]["properties"]
            properties["text"]["pattern"] = bounded_text_pattern(limits["text_units"])
            properties["evidence_event_ids"]["maxItems"] = limits["references_per_claim"]
    return schema


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate structured draft keys")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ValueError("Structured draft is not strict JSON")


def _validate_serialized_text_limits(raw_text: str, units: int) -> None:
    """Check original text lexemes after strict JSON parsing, without normalizing escapes."""
    decoder, cursor = json.JSONDecoder(), 0
    pattern = re.compile(bounded_text_pattern(units))
    while (start := raw_text.find('"', cursor)) != -1:
        value, end = decoder.raw_decode(raw_text, start)
        cursor = end
        after = end
        while after < len(raw_text) and raw_text[after] in " \t\r\n":
            after += 1
        if value != "text" or after == len(raw_text) or raw_text[after] != ":":
            continue
        start = after + 1
        while start < len(raw_text) and raw_text[start] in " \t\r\n":
            start += 1
        value, cursor = decoder.raw_decode(raw_text, start)
        # Shape validation below diagnoses non-string claim values separately.
        if isinstance(value, str) and not pattern.fullmatch(raw_text[start + 1:cursor - 1]):
            raise ValueError("Structured bounded draft limits exceeded")


def assemble_structured_summary(
    raw_text: str, visible_ids: set[str], *, token_budget: int | None = None,
) -> str:
    """Append only model-selected citations; preserve claim text and never infer evidence."""
    try:
        data = json.loads(
            raw_text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except json.JSONDecodeError as exc:
        raise ValueError("Summary is not valid JSON") from exc
    if not isinstance(data, dict) or set(data) != set(CLAIM_FIELDS):
        raise ValueError("Structured draft fields do not match schema")
    limits = draft_limits(token_budget) if token_budget is not None else None
    if limits:
        _validate_serialized_text_limits(raw_text, limits["text_units"])
    cited, assembled = set(), {}
    for field in CLAIM_FIELDS:
        claims = data[field]
        if not isinstance(claims, list):
            raise ValueError("Structured draft claims must be arrays of citation objects")
        if limits and len(claims) > limits["claims_per_field"]:
            raise ValueError("Structured bounded draft limits exceeded")
        assembled[field] = []
        for claim in claims:
            if not isinstance(claim, dict) or set(claim) != {"text", "evidence_event_ids"}:
                raise ValueError("Structured draft claim fields do not match schema")
            text, selected = claim["text"], claim["evidence_event_ids"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Structured draft claims require nonempty text")
            if limits:
                try:
                    text.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise ValueError("Structured draft contains invalid Unicode") from exc
            if not isinstance(selected, list) or not all(
                isinstance(item, str) for item in selected
            ):
                raise ValueError("Invalid source_event_ids")
            if not selected:
                raise ValueError("Each structured claim requires a visible event citation")
            if limits and (
                len(text) > limits["text_units"]
                or len(selected) > limits["references_per_claim"]
            ):
                raise ValueError("Structured bounded draft limits exceeded")
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
