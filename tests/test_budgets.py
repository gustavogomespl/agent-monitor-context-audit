import pytest

from context_audit.representations import budget_for, head_tail, validate_summary


def test_budget_rule_is_bounded_and_does_not_fake_compression():
    assert [budget_for(n) for n in (0, 20, 128, 400, 1000, 8000)] == [0, 20, 128, 128, 250, 1024]
    with pytest.raises(ValueError):
        budget_for(-1)


def test_head_tail_accounts_for_marker_and_keeps_both_ends():
    body = "A" * 400 + "M" * 400 + "Z" * 400
    text = head_tail(body, 128, len)
    assert len(text) <= 128
    assert text.startswith("AAAA") and text.endswith("ZZZZ")
    assert "omitted" in text and "MMMM" not in text
    assert head_tail("short", 5, len) == "short"


def test_summary_rejects_bad_json_unknown_evidence_and_overbudget():
    assert validate_summary("Result observed [E0001].", "free_summary", 100, len, {"E0001"})
    with pytest.raises(ValueError):
        validate_summary("Result [E9999]", "free_summary", 100, len, {"E0001"})
    with pytest.raises(ValueError):
        validate_summary("No citations", "free_summary", 100, len, {"E0001"})
    with pytest.raises(ValueError):
        validate_summary("{broken", "structured_summary", 100, len, {"E0001"})
    with pytest.raises(ValueError):
        validate_summary("x" * 101, "free_summary", 100, len, set())
