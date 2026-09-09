"""Synthetic v7 prose/citation separation; historical formats remain unchanged."""

import copy
import json
import re

import pytest

from context_audit.structured_summary import (
    CLAIM_FIELDS,
    assemble_structured_summary,
    structured_summary_schema,
)


def draft(text="The independent operation completed.", selected=None):
    value = {field: [] for field in CLAIM_FIELDS}
    value[CLAIM_FIELDS[0]] = [{
        "text": text,
        "evidence_event_ids": ["E0003"] if selected is None else selected,
    }]
    return value


def assemble(raw, **options):
    return assemble_structured_summary(
        raw, {"E0003", "E0004"}, token_budget=2048, text_bounds=False,
        citations_in_text=False, **options,
    )


def schema():
    return structured_summary_schema(
        {"E0003", "E0004"}, token_budget=2048, text_bounds=False,
        citations_in_text=False,
    )


def test_separation_changes_only_prose_grammar_and_keeps_structural_caps():
    original = structured_summary_schema(
        {"E0003", "E0004"}, token_budget=2048, text_bounds=False,
    )
    actual = schema()
    stripped = copy.deepcopy(actual)
    patterns = []
    for field in CLAIM_FIELDS:
        claim = stripped["properties"][field]["items"]["properties"]
        patterns.append(claim["text"].pop("pattern"))
        assert stripped["properties"][field]["maxItems"] == 2
        assert claim["evidence_event_ids"]["maxItems"] == 2
    assert len(set(patterns)) == 1
    assert stripped == original
    assert structured_summary_schema(
        {"E0003", "E0004"}, token_budget=2048, text_bounds=False,
    ) == original


@pytest.mark.parametrize("text", [
    "E0003", "Observed [E0003].", "Observed E0003/E0004.",
    "A later E00003 event.", "E１２３４", "E٣٢١٤", "E0003😀",
    "E0003\n", "E0003\u0301", "E0003 E1", "E0003\\path",
])
def test_decoded_inline_references_fail_even_when_selected(text):
    raw = json.dumps(draft(text), ensure_ascii=False)
    with pytest.raises(ValueError, match="Structured claim text must not contain event IDs"):
        assemble(raw)


@pytest.mark.parametrize("ensure_ascii", [True, False])
def test_unicode_escape_cannot_hide_an_inline_reference(ensure_ascii):
    raw = json.dumps(draft("E0003"), ensure_ascii=ensure_ascii)
    for encoded in [r"\u00450003", r"E\u0030003", r"\u0045\u0030\u0030\u0030\u0033"]:
        with pytest.raises(ValueError, match="Structured claim text must not contain event IDs"):
            assemble(raw.replace(
                '"text": "E0003"', '"text": "' + encoded + '"',
            ))


@pytest.mark.parametrize("text", [
    'Exact "quote", \\ path, café\n雪 🧪. ', "CODE1234", "E123", "E0003suffix",
    "éE0003", "E0003é", "E0003_", "123E0003", "E²³¹⁴", "e0003",
    "Literal text key: \"text\": \"preserve it\".", "Long ordinary prose. " * 60,
    r"The literal sequence \u0045 is retained.", "E0003漢字", "漢字E0003",
    "𱋥E0003", "E0003𱋥", "흍E0003", "E0003흍",
])
def test_normal_identifiers_unicode_and_long_prose_are_preserved(text):
    raw = json.dumps(draft(text, ["E0004", "E0003"]), ensure_ascii=False)
    final = json.loads(assemble(raw))
    assert final[CLAIM_FIELDS[0]] == [text + " [E0003] [E0004]"]
    assert final["source_event_ids"] == ["E0003", "E0004"]
    pattern = schema()["properties"][CLAIM_FIELDS[0]]["items"]["properties"]["text"]["pattern"]
    assert re.fullmatch(pattern, json.dumps(text, ensure_ascii=False)[1:-1])


def test_canonical_lexeme_rule_keeps_control_escapes_but_does_not_rewrite_unicode():
    raw = json.dumps(draft("A\x00\x1f\nB"), ensure_ascii=False)
    assert json.loads(assemble(raw))[CLAIM_FIELDS[0]] == ["A\x00\x1f\nB [E0003]"]
    raw = json.dumps(draft("café 🧪"), ensure_ascii=True)
    with pytest.raises(ValueError, match="literal non-control Unicode"):
        assemble(raw)
    # The earlier mode continues to accept either JSON encoding.
    assert assemble_structured_summary(
        raw, {"E0003", "E0004"}, token_budget=2048, text_bounds=False,
    )


@pytest.mark.parametrize("mutation", ["unknown", "missing", "many_claims", "many_refs"])
def test_separation_never_repairs_citations_or_drops_claims(mutation):
    value = draft()
    claim = value[CLAIM_FIELDS[0]][0]
    if mutation == "unknown":
        claim["evidence_event_ids"] = ["E0999"]
    elif mutation == "missing":
        claim["evidence_event_ids"] = []
    elif mutation == "many_claims":
        value[CLAIM_FIELDS[0]] *= 3
    else:
        claim["evidence_event_ids"] *= 3
    raw = json.dumps(value)
    with pytest.raises(ValueError):
        assemble(raw)
    assert json.dumps(value) == raw


def test_historical_inline_citations_are_still_valid():
    raw = json.dumps(draft("Observed [E0003]."))
    for options in [{}, {"token_budget": 2048}, {"token_budget": 2048, "text_bounds": False}]:
        assert assemble_structured_summary(raw, {"E0003"}, **options)


def test_separation_cannot_silently_replace_historical_character_bounds():
    with pytest.raises(ValueError, match="require compact prose"):
        structured_summary_schema({"E0003"}, token_budget=2048, citations_in_text=False)
    with pytest.raises(ValueError, match="require compact prose"):
        assemble_structured_summary(
            json.dumps(draft()), {"E0003"}, token_budget=2048, citations_in_text=False,
        )


def test_actual_xgrammar_prose_semantics_when_optional_backend_is_installed():
    xgr = pytest.importorskip("xgrammar")
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo([]))
    compiled = compiler.compile_json_schema(schema())
    for text in [
        "CODE1234", "E0003suffix", "café 🧪", 'a "quote" \\ path\nline', "éE0003",
        "𱋥E0003", "E0003𱋥", "흍E0003", "E0003흍",
    ]:
        assert xgr.GrammarMatcher(compiled).accept_string(
            json.dumps(draft(text), ensure_ascii=False),
        )
    for text in ["[E0003]", "E0003", "E１２３４", "E0003😀"]:
        assert not xgr.GrammarMatcher(compiled).accept_string(
            json.dumps(draft(text), ensure_ascii=False),
        )
    for encoded in [r"\u00450003", r"E\u0030003", r"\ud83e\uddea"]:
        raw = json.dumps(draft()).replace(
            '"The independent operation completed."', '"' + encoded + '"',
        )
        assert not xgr.GrammarMatcher(compiled).accept_string(raw)
