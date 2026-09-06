"""Bounded representations; all text is inert data, never executable code."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable

Count = Callable[[str], int]
CONDITIONS = ("full", "head_tail", "free_summary", "structured_summary")
OMISSION = "\n[... content omitted; boundary fragments may be incomplete ...]\n"
STRUCTURED_FIELDS = {
    "environment_and_state",
    "actions_and_observed_results",
    "important_identifiers",
    "contradictions_or_missing_information",
    "source_event_ids",
}


def budget_for(tokens: int, fraction: float = 0.25, minimum: int = 128, maximum: int = 1024) -> int:
    if tokens < 0 or not 0 < fraction <= 1 or minimum < 1 or maximum < minimum:
        raise ValueError("Invalid token budget parameters")
    return min(tokens, max(minimum, min(maximum, math.floor(fraction * tokens))))


def _fit(text: str, budget: int, count: Count, *, tail: bool = False) -> str:
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        chunk = text[-mid:] if tail else text[:mid]
        if count(chunk) <= budget:
            low = mid
        else:
            high = mid - 1
    return (text[-low:] if tail else text[:low]) if low else ""


def head_tail(body: str, budget: int, count: Count) -> str:
    if count(body) <= budget:
        return body
    if count(OMISSION) > budget:
        raise ValueError("Budget cannot represent the omission marker")
    available = budget - count(OMISSION)
    head = _fit(body, available // 2, count)
    tail = _fit(body, available - available // 2, count, tail=True)
    # Tokenization is not additive/strictly monotone. Validate the assembled text.
    while count(head + OMISSION + tail) > budget:
        if not head and not tail:
            raise ValueError("Omission marker exceeds measured token budget")
        head = head[: -max(1, len(head) // 20)] if head else ""
        tail = tail[max(1, len(tail) // 20) :] if tail else ""
    return head + OMISSION + tail


def evidence_ids(text: str) -> set[str]:
    return set(re.findall(r"\bE\d{4,}\b", text))


def validate_summary(
    text: str, condition: str, budget: int, count: Count, visible_ids: set[str]
) -> str:
    if count(text) > budget:
        raise ValueError("Summary exceeds token budget")
    if not text.strip():
        raise ValueError("Empty summary")
    if condition == "structured_summary":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Summary is not valid JSON") from exc
        if not isinstance(data, dict) or set(data) != STRUCTURED_FIELDS:
            raise ValueError("Structured summary fields do not match schema")
        for key in STRUCTURED_FIELDS - {"source_event_ids"}:
            if not isinstance(data[key], list) or not all(isinstance(v, str) for v in data[key]):
                raise ValueError("Structured claims must be arrays of cited strings")
            for claim in data[key]:
                if not evidence_ids(claim):
                    raise ValueError("Each structured claim requires a visible event citation")
        if not isinstance(data["source_event_ids"], list) or not all(
            isinstance(v, str) and re.fullmatch(r"E\d{4,}", v) for v in data["source_event_ids"]
        ):
            raise ValueError("Invalid source_event_ids")
        cited_claims = evidence_ids(
            json.dumps({k: v for k, v in data.items() if k != "source_event_ids"})
        )
        if cited_claims != set(data["source_event_ids"]):
            raise ValueError("Source IDs must match IDs cited in claims")
    elif condition != "free_summary":
        raise ValueError("Unknown summary condition")
    cited = evidence_ids(text)
    if not cited <= visible_ids or (visible_ids and not cited):
        raise ValueError("Missing or unknown source event IDs")
    return text
