"""Synthetic regression coverage for the v6 decoder and unchanged final budget."""

import copy
import json

import pytest
from test_structured_summary_v3 import SyntheticProvider, config, draft, synthetic_transcript

from context_audit.runner import make_representation, monitor_representation
from context_audit.storage import PrivateStore
from context_audit.structured_summary import (
    CLAIM_FIELDS,
    assemble_structured_summary,
    structured_summary_schema,
)


def compact_config():
    return config("schema_citations_compact_v1").model_copy(update={
        "token_minimum": 1024, "token_maximum": 2048, "summary_max_tokens": 3200,
        "monitor_output_mode": "schema_visible_evidence_v1",
    })


def test_compact_decoder_removes_only_expensive_text_constraints():
    visible = {"E0002", "E0003"}
    old = structured_summary_schema(visible, token_budget=2048)
    expected = copy.deepcopy(old)
    for field in CLAIM_FIELDS:
        expected["properties"][field]["items"]["properties"]["text"].pop("pattern")
    assert structured_summary_schema(visible, token_budget=2048, text_bounds=False) == expected
    assert structured_summary_schema(visible, token_budget=2048) == old


def test_compact_assembly_keeps_long_claim_whole_when_final_budget_allows(tmp_path):
    fields = draft()
    text = "Independent observation. " * 26
    fields[CLAIM_FIELDS[0]][0]["text"] = text
    raw = json.dumps(fields)
    with pytest.raises(ValueError, match="bounded draft"):
        assemble_structured_summary(raw, {"E0002", "E0003"}, token_budget=2048)
    store = PrivateStore(tmp_path)
    provider = SyntheticProvider(store, [raw])
    rep, calls = make_representation(
        synthetic_transcript(), "structured_summary", compact_config(), provider, store, 0, 196608,
    )
    assert rep.status == "ok" and len(calls) == 1
    assert json.loads(rep.text)[CLAIM_FIELDS[0]] == [text + " [E0002]"]
    assert "JSON text units" not in provider.requests[0]["text"]
    assert "at most 2" in provider.requests[0]["text"]
    assert "pattern" not in json.dumps(provider.requests[0]["structured_schema"])


@pytest.mark.parametrize("mutation", ["too_many_claims", "too_many_references", "too_many_tokens"])
def test_compact_mode_still_rejects_overflow_without_cutting_or_monitoring(tmp_path, mutation):
    fields = draft()
    if mutation == "too_many_claims":
        fields[CLAIM_FIELDS[0]] *= 3
    elif mutation == "too_many_references":
        fields[CLAIM_FIELDS[0]][0]["evidence_event_ids"] = ["E0002"] * 3
    else:
        fields[CLAIM_FIELDS[0]][0]["text"] = "x" * 2100
    store = PrivateStore(tmp_path)
    provider = SyntheticProvider(store, [json.dumps(fields)] * 2)
    cfg, transcript = compact_config(), synthetic_transcript()
    rep, calls = make_representation(
        transcript, "structured_summary", cfg, provider, store, 0, 196608,
    )
    assert rep.status == ("budget_violation" if mutation == "too_many_tokens" else "invalid_output")
    assert rep.text == "" and len(calls) == 2
    result, monitor_calls = monitor_representation(transcript, rep, cfg, provider, 0, 196608)
    assert result.suspicion_score is None and result.escalate and not monitor_calls


def test_monitor_schema_uses_compressed_visible_ids_on_both_attempts(tmp_path):
    from context_audit.render import render_body
    from context_audit.runtime_models import Representation
    from context_audit.storage import digest

    transcript, cfg = synthetic_transcript(), compact_config()
    text = "Observed toy event [E0002]."
    rep = Representation(
        transcript_id=transcript.transcript_id, condition="head_tail", text=text,
        budget=1024, body_tokens=len(render_body(transcript)), measured_tokens=len(text),
        status="ok", content_hash=digest(text), data_origin="synthetic_fixture",
        counter_method="synthetic_character_count", compression_applied=True,
    )
    decision = {
        "suspicion_score": 12.5, "brief_justification": "Independent observation.",
        "evidence_event_ids": ["E0003"], "uncertainties": [],
    }
    store = PrivateStore(tmp_path)
    provider = SyntheticProvider(store, [json.dumps(decision)] * 2)
    result, calls = monitor_representation(transcript, rep, cfg, provider, 0, 196608)
    assert result.status == "invalid_output" and result.suspicion_score is None and len(calls) == 2
    for request in provider.requests:
        schema = request["structured_schema"]
        assert schema["properties"]["evidence_event_ids"]["items"]["enum"] == ["E0002"]
        assert "E0003" not in request["text"]
    assert provider.requests[0]["structured_schema"] == provider.requests[1]["structured_schema"]


def test_monitor_schema_is_qwen_only_and_legacy_default_remains_prompt():
    from context_audit.runtime_models import AuditConfig

    assert AuditConfig().monitor_output_mode == "prompt"
    with pytest.raises(ValueError, match="Qwen"):
        AuditConfig(monitor_output_mode="schema_visible_evidence_v1")
