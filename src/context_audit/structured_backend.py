"""CPU-only verification of the pinned citation decoder before model startup."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json

from context_audit.structured_summary import CLAIM_FIELDS, draft_limits, structured_summary_schema

XGRAMMAR_VERSION = "0.2.3"


def _check_separate_ids(xgr, compiler) -> dict:
    """Falsify inline-citation and escape bypasses using independent complete JSON."""
    schema = structured_summary_schema(
        {"E0003", "E0004"}, token_budget=2048, text_bounds=False, citations_in_text=False,
    )
    compiled = compiler.compile_json_schema(schema)
    accepted, rejected = 0, 0

    def draft(text):
        return {
            field: [{"text": text, "evidence_event_ids": ["E0003"]}] if index == 0 else []
            for index, field in enumerate(CLAIM_FIELDS)
        }

    def check(raw, expected):
        nonlocal accepted, rejected
        matcher = xgr.GrammarMatcher(compiled)
        actual = matcher.accept_string(raw) and matcher.is_completed()
        if actual != expected:
            raise ValueError("Separate-ID decoder failed an independent schema check")
        accepted += int(expected)
        rejected += int(not expected)

    for text in (
        'Exact "quote", \\ path, café\n雪 🧪.', "CODE1234", "E0003suffix",
        "éE0003", "E0003é", "E0003_", "E123", "E²³¹⁴", "A\x00\x1fB",
        "Independent long observation. " * 60, "Exact \U000312e5 identifier.",
        "𱋥E0003", "E0003𱋥", "흍E0003", "E0003흍",
    ):
        check(json.dumps(draft(text), ensure_ascii=False), True)
    for text in (
        "E0003", "Observed [E0003].", "E0003/E0004", "E00003", "E１２３４",
        "E٣٢١٤", "E0003😀", "E0003\n", "E0003\u0301", "E0003\\path",
    ):
        check(json.dumps(draft(text), ensure_ascii=False), False)
    # Non-control Unicode must be literal in this mode. Neither an escaped E
    # nor an escaped digit can turn a forbidden reference into permitted prose.
    for encoded in (r"\u00450003", r"E\u0030003", r"caf\u00e9", r"\ud83e\uddea"):
        raw = json.dumps(draft("placeholder")).replace(
            '"placeholder"', '"' + encoded + '"',
        )
        check(raw, False)
    for selected in ([], ["E9999"], ["E0003"] * 3):
        example = draft("Independent observation.")
        example[CLAIM_FIELDS[0]][0]["evidence_event_ids"] = selected
        check(json.dumps(example), False)
    return {
        "accepted_cases": accepted, "rejected_cases": rejected,
        "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
        "model_generation_executed": False,
    }


def _check_compact_and_monitor(xgr, compiler) -> dict:
    """Verify the v6 schemas' evidence and shape rules without a model call."""
    from context_audit.monitor_schema import monitor_response_schema

    visible = {"E0003", "E0004"}
    schemas = {
        "compact": structured_summary_schema(visible, token_budget=2048, text_bounds=False),
        "monitor": monitor_response_schema(visible),
        "empty_monitor": monitor_response_schema(set()),
    }
    compiled = {name: compiler.compile_json_schema(schema) for name, schema in schemas.items()}
    accepted, rejected = 0, 0

    def check(name, body, expected):
        nonlocal accepted, rejected
        matcher = xgr.GrammarMatcher(compiled[name])
        actual = (
            matcher.accept_string(json.dumps(body, ensure_ascii=True)) and matcher.is_completed()
        )
        if actual != expected:
            raise ValueError("Compact/monitor decoder failed an independent schema check")
        accepted += int(expected)
        rejected += int(not expected)

    claim = {"text": 'Exact "quote", \\ path, café\n😀. ' * 30,
             "evidence_event_ids": ["E0003"]}
    draft = {field: [copy.deepcopy(claim)] for field in CLAIM_FIELDS}
    check("compact", draft, True)
    for invalid in (
        [{"text": "No references."}],
        [{**claim, "evidence_event_ids": []}],
        [{**claim, "evidence_event_ids": ["E9999"]}],
        [{**claim, "evidence_event_ids": ["E0003"] * 3}],
        [claim] * 3,
    ):
        example = copy.deepcopy(draft)
        example[CLAIM_FIELDS[0]] = invalid
        check("compact", example, False)
    monitor = {
        "suspicion_score": 12.5, "brief_justification": claim["text"],
        "evidence_event_ids": ["E0003"], "uncertainties": [],
    }
    check("monitor", monitor, True)
    check("monitor", {**monitor, "evidence_event_ids": []}, True)
    check("empty_monitor", {**monitor, "evidence_event_ids": []}, True)
    check("empty_monitor", monitor, False)
    for change in (
        {"evidence_event_ids": ["E9999"]}, {"suspicion_score": 101},
        {"suspicion_score": -1}, {"suspicion_score": "12.5"}, {"extra": "invalid"},
    ):
        check("monitor", {**monitor, **change}, False)
    missing = dict(monitor)
    del missing["uncertainties"]
    check("monitor", missing, False)
    return {
        "accepted_cases": accepted, "rejected_cases": rejected,
        "schema_hashes": {
            name: hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
            for name, schema in schemas.items()
        },
        "model_generation_executed": False,
    }


def _check_bounded_schema(xgr, compiler, token_budget: int) -> dict:
    """Falsify structural decoder bounds with fixed, independent synthetic strings."""
    schema = structured_summary_schema({"E0003", "E0004"}, token_budget=token_budget)
    compiled = compiler.compile_json_schema(schema)
    limits = draft_limits(token_budget)
    units = limits["text_units"]
    accepted, rejected = 0, 0

    def check_raw(raw: str, expected: bool, case: str):
        nonlocal accepted, rejected
        matcher = xgr.GrammarMatcher(compiled)
        actual = matcher.accept_string(raw) and matcher.is_completed()
        if actual != expected:
            action = "rejected" if expected else "accepted"
            raise ValueError(f"Bounded decoder {action} synthetic {case} at budget {token_budget}")
        if expected:
            accepted += 1
        else:
            rejected += 1

    def draft(text: str) -> dict:
        return {
            field: [{"text": text, "evidence_event_ids": ["E0003"]}] if index == 0 else []
            for index, field in enumerate(CLAIM_FIELDS)
        }

    def check(body: dict, expected: bool, case: str, *, ensure_ascii: bool = True):
        check_raw(json.dumps(body, ensure_ascii=ensure_ascii), expected, case)

    mixed = 'Toy "quote", backslash \\, newline\nand unicode: café 😀.'
    for case, text, ascii_only in (
        ("ordinary text", "Independent toy observation.", True),
        ("mixed escaped text", mixed, True),
        ("mixed literal unicode", mixed, False),
        ("ASCII boundary", "a" * units, True),
        ("escaped unicode boundary", "é" * units, True),
        ("literal emoji boundary", "😀" * units, False),
        ("escaped quote boundary", '"' * units, True),
        # Each UTF-16 surrogate escape consumes one grammar unit.
        ("surrogate pair boundary", "😀" * (units // 2), True),
    ):
        check(draft(text), True, case, ensure_ascii=ascii_only)
    maximum = {
        field: [
            {"text": "a" * units, "evidence_event_ids": ["E0003", "E0004"]}
            for _ in range(2)
        ]
        for field in CLAIM_FIELDS
    }
    check(maximum, True, "all structural boundaries")

    for case, text, ascii_only in (
        ("ASCII length overflow", "a" * (units + 1), True),
        ("escaped unicode overflow", "é" * (units + 1), True),
        ("literal emoji overflow", "😀" * (units + 1), False),
        ("escaped quote overflow", '"' * (units + 1), True),
        ("surrogate pair overflow", "😀" * (units // 2 + 1), True),
    ):
        check(draft(text), False, case, ensure_ascii=ascii_only)
    too_many_claims = draft("toy")
    too_many_claims[CLAIM_FIELDS[0]] *= 3
    check(too_many_claims, False, "claim count overflow")
    too_many_references = draft("toy")
    too_many_references[CLAIM_FIELDS[0]][0]["evidence_event_ids"] = [
        "E0003", "E0004", "E0003",
    ]
    check(too_many_references, False, "reference count overflow")

    cited = {"text": "toy", "evidence_event_ids": ["E0003"]}
    for invalid in (
        {"text": "orphan"},
        {"text": "orphan", "evidence_event_ids": []},
        {"text": "orphan", "evidence_event_ids": ["E9999"]},
    ):
        for neighbors in ([invalid, cited], [cited, invalid]):
            example = draft("toy")
            example[CLAIM_FIELDS[0]] = neighbors
            check(example, False, "invalid item beside a cited neighbor")

    for content in ('raw"quote', "raw\nnewline", "raw\x00null", r"invalid\qescape"):
        raw = json.dumps(draft("toy")).replace('"toy"', '"' + content + '"')
        check_raw(raw, False, "unescaped delimiter/control or invalid escape")
    # The first item's escaped quote/backslash/unicode must never let its text
    # consume the next object's citation, as a permissive string regex could.
    for raw_text in (r'"\""', r'"\\"', r'"\u0022"'):
        example = draft("toy")
        example[CLAIM_FIELDS[0]] = [{"text": "orphan"}, cited]
        raw = json.dumps(example).replace('"orphan"', raw_text)
        check_raw(raw, False, "escaped delimiter beside a cited neighbor")
    missing = draft("toy")
    missing.pop(CLAIM_FIELDS[-1])
    check(missing, False, "missing summary field")
    return {
        "status": "passed", "token_budget": token_budget, "limits": limits,
        "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
        "accepted_cases": accepted, "rejected_cases": rejected,
        "exact_token_ceiling_verified": False,
    }


def check_structured_backend(*, include_separate_ids: bool = False) -> dict:
    """Compile the actual schema and test complete strings using independent toy inputs."""
    version = importlib.metadata.version("xgrammar")
    if version != XGRAMMAR_VERSION:
        raise ValueError(f"Citation decoding requires xgrammar=={XGRAMMAR_VERSION}")
    import xgrammar as xgr

    schema = structured_summary_schema({"E0003", "E0004"})
    grammar = xgr.Grammar.from_json_schema(schema)
    # A small byte vocabulary is enough to validate grammar matching, without any
    # model tokenizer download, GPU use or generation. Colab uses the same compiler.
    vocabulary = [chr(value) for value in range(32, 127)] + ["<eos>"]
    info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[len(vocabulary) - 1])
    compiler = xgr.GrammarCompiler(info, max_threads=1)
    compiled = compiler.compile_grammar(grammar)

    def accepts(body):
        matcher = xgr.GrammarMatcher(compiled)
        text = json.dumps(body, ensure_ascii=True)
        return matcher.accept_string(text) and matcher.is_completed()

    cited = {"text": "Independent toy observation.", "evidence_event_ids": ["E0003"]}
    valid = {field: [copy.deepcopy(cited)] for field in CLAIM_FIELDS}
    if not accepts(valid):
        raise ValueError("Citation decoder rejected a valid synthetic summary")
    escaped = copy.deepcopy(valid)
    escaped[CLAIM_FIELDS[0]][0]["text"] = 'Toy "quote", newline\nand unicode: café.'
    if not accepts(escaped):
        raise ValueError("Citation decoder rejected valid escaped claim text")
    rejected = 0
    for field in CLAIM_FIELDS:
        for invalid in (
            {"text": "No references field."},
            {"text": "Empty references.", "evidence_event_ids": []},
            {"text": "Unknown reference.", "evidence_event_ids": ["E9999"]},
            {"evidence_event_ids": ["E0003"]},
            {**cited, "label": "synthetic forbidden field"},
        ):
            # A cited neighbor must never satisfy another item's missing citation.
            for pair in ([invalid, cited], [cited, invalid]):
                example = copy.deepcopy(valid)
                example[field] = pair
                if accepts(example):
                    raise ValueError("Citation decoder accepted an invalid synthetic item")
                rejected += 1
    missing = copy.deepcopy(valid)
    missing.pop(CLAIM_FIELDS[-1])
    if accepts(missing):
        raise ValueError("Citation decoder accepted a missing summary field")
    rejected += 1
    receipt = {
        "status": "passed", "backend": "xgrammar", "version": version,
        "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
        "accepted_cases": 2, "rejected_cases": rejected, "model_generation_executed": False,
        "bounded_checks": [_check_bounded_schema(xgr, compiler, budget) for budget in (1024, 2048)],
        "compact_and_monitor_checks": _check_compact_and_monitor(xgr, compiler),
    }
    if include_separate_ids:
        receipt["separate_ids_checks"] = _check_separate_ids(xgr, compiler)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--separate-ids", action="store_true")
    args = parser.parse_args()
    print(json.dumps(
        check_structured_backend(include_separate_ids=args.separate_ids), sort_keys=True,
    ))
