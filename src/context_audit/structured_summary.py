"""Constrained summary drafts and a deterministic index of existing citations."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from functools import cache

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


def _unicode_class(predicate: Callable[[str], bool]) -> str:
    """Explicit positive ranges avoid XGrammar 0.2.3's Unicode-complement bug."""
    def escaped(value: int) -> str:
        if value <= 0xff:
            return rf"\x{value:02x}"
        if value <= 0xffff:
            return rf"\u{value:04x}"
        return rf"\U{value:08x}"

    def append_range(start: int, end: int) -> None:
        # XGrammar 0.2.3 also mishandles UTF-8 ranges across some leading-byte
        # boundaries (including Hangul and supplementary CJK). Small explicit
        # blocks preserve both ends instead of normalizing away valid scalars.
        while start <= end:
            block = 128 if start < 128 else 64 if start < 0x800 else 4096
            boundary = ((start // block) + 1) * block - 1
            stop = min(end, boundary)
            ranges.append(escaped(start) if start == stop else escaped(start) + "-" + escaped(stop))
            start = stop + 1

    ranges, start, end = [], None, None
    for value in range(0x110000):
        if not predicate(chr(value)):
            continue
        if start is not None and value == end + 1:
            end = value
            continue
        if start is not None:
            append_range(start, end)
        start = end = value
    if start is not None:
        append_range(start, end)
    return "".join(ranges)


@cache
def prose_without_citations_pattern() -> str:
    r"""Unbounded finite-state prose excluding the decoded ``\bE\d{4,}\b`` language.

    Python's Unicode word/digit classes preserve identifiers such as CODE1234,
    E0003suffix and éE0003. No length state is introduced. JSON string patterns
    in XGrammar 0.2.3 operate on serialized lexemes, so non-control Unicode must
    be emitted literally: Unicode escapes are permitted only for U+0000..001F.
    Quotes, backslashes, slashes and ordinary control escapes remain available.
    This canonical encoding also prevents escaped E/digits bypassing the grammar.
    """
    def word(char: str) -> bool:
        return char.isalnum() or char == "_"

    words = _unicode_class(word)
    digits = _unicode_class(str.isdecimal)
    not_e = _unicode_class(lambda char: word(char) and char != "E")
    not_digit = _unicode_class(lambda char: word(char) and not char.isdecimal())
    nonwords = _unicode_class(lambda char: (
        not word(char) and ord(char) >= 32 and char not in '"\\'
        and not 0xd800 <= ord(char) <= 0xdfff
    ))
    separator = rf'([{nonwords}]|\\["\\/bfnrt]|\\u00[01][0-9a-fA-F])'
    allowed_word = (
        rf'([{not_e}][{words}]*|E([{digits}]{{0,3}}|[{digits}]*[{not_digit}][{words}]*))'
    )
    # Every completed word must be followed by a non-word unit or the end; words
    # cannot be split into independently allowed fragments such as E000 + 3.
    return rf'^({separator}|{allowed_word}{separator})*({allowed_word})?$'


def structured_summary_schema(
    visible_ids: set[str], *, token_budget: int | None = None, text_bounds: bool = True,
    citations_in_text: bool = True,
) -> dict:
    """Return an independent schema; its claims, never an index, are model-generated."""
    if not visible_ids or any(
        not isinstance(event_id, str) or not re.fullmatch(r"E[0-9]{4,}", event_id)
        for event_id in visible_ids
    ):
        raise ValueError("Structured generation requires visible event IDs")
    if not citations_in_text and text_bounds:
        raise ValueError("Separate citations require compact prose without character bounds")
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
            if text_bounds:
                properties["text"]["pattern"] = bounded_text_pattern(limits["text_units"])
            properties["evidence_event_ids"]["maxItems"] = limits["references_per_claim"]
    if not citations_in_text:
        pattern = prose_without_citations_pattern()
        for field in CLAIM_FIELDS:
            schema["properties"][field]["items"]["properties"]["text"]["pattern"] = pattern
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


def _validate_separate_text_encoding(raw_text: str) -> None:
    """Reject noncanonical lexemes without reserializing or repairing claim text."""
    decoder, cursor = json.JSONDecoder(), 0
    pattern = re.compile(prose_without_citations_pattern())
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
        if isinstance(value, str) and not pattern.fullmatch(raw_text[start + 1:cursor - 1]):
            raise ValueError("Structured claim text must use literal non-control Unicode")


def assemble_structured_summary(
    raw_text: str, visible_ids: set[str], *, token_budget: int | None = None,
    text_bounds: bool = True, citations_in_text: bool = True,
) -> str:
    """Append only model-selected citations; preserve claim text and never infer evidence."""
    if not citations_in_text and text_bounds:
        raise ValueError("Separate citations require compact prose without character bounds")
    try:
        data = json.loads(
            raw_text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except json.JSONDecodeError as exc:
        raise ValueError("Summary is not valid JSON") from exc
    if not isinstance(data, dict) or set(data) != set(CLAIM_FIELDS):
        raise ValueError("Structured draft fields do not match schema")
    limits = draft_limits(token_budget) if token_budget is not None else None
    if limits and text_bounds:
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
                (text_bounds and len(text) > limits["text_units"])
                or len(selected) > limits["references_per_claim"]
            ):
                raise ValueError("Structured bounded draft limits exceeded")
            selected = set(selected)
            mentioned = evidence_ids(text)
            if not citations_in_text and mentioned:
                raise ValueError("Structured claim text must not contain event IDs")
            if not (selected | mentioned) <= visible_ids:
                raise ValueError("Missing or unknown source event IDs")
            if not mentioned <= selected:
                raise ValueError("Claim text references event IDs not selected for this item")
            cited.update(selected)
            assembled[field].append(text + " " + " ".join(f"[{item}]" for item in sorted(selected)))
    if visible_ids and not cited:
        raise ValueError("Missing or unknown source event IDs")
    if not citations_in_text:
        _validate_separate_text_encoding(raw_text)
    assembled["source_event_ids"] = sorted(cited)
    return json.dumps(assembled, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
