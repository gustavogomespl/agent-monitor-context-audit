"""Synthetic runner checks for generation with citations in one dedicated field."""

import json

import pytest
from test_compact_summary_v6 import compact_config
from test_structured_summary_v3 import SyntheticProvider, draft, synthetic_transcript

from context_audit.runner import make_representation, monitor_representation, prompts
from context_audit.runtime_models import AuditConfig
from context_audit.storage import PrivateStore


def separate_config():
    return AuditConfig.model_validate(compact_config().model_dump() | {
        "structured_summary_mode": "schema_citations_separate_ids_v1",
        "protocol_version": "development-summary-v7",
    })


def test_separate_ids_request_preserves_identifier_and_selected_evidence(tmp_path):
    cfg = separate_config()
    fields = draft()
    text = 'Exact CODE1234, café, "quote" and \\ path.'
    fields["important_identifiers"][0]["text"] = text
    store = PrivateStore(tmp_path)
    provider = SyntheticProvider(store, [json.dumps(fields)])
    rep, calls = make_representation(
        synthetic_transcript(), "structured_summary", cfg, provider, store, 0, 196608,
    )
    assert rep.status == "ok" and len(calls) == 1
    assert json.loads(rep.text)["important_identifiers"] == [text + " [E0002]"]
    request = provider.requests[0]
    assert request["system"] != prompts(compact_config())["summary_common"] + "\n\n" + (
        prompts(compact_config())["summary_structured"]
    )
    assert request["structured_schema"]["properties"]["important_identifiers"]["items"][
        "properties"
    ]["text"]["pattern"]
    assert request["max_tokens"] == 3200


def test_separate_ids_retries_original_input_without_repairing_or_reusing_bad_text(tmp_path):
    bad = draft()
    bad["important_identifiers"][0]["text"] = "SYNTHETIC_REJECTED_PAYLOAD E0002."
    store = PrivateStore(tmp_path)
    provider = SyntheticProvider(store, [json.dumps(bad), json.dumps(draft())])
    rep, calls = make_representation(
        synthetic_transcript(), "structured_summary", separate_config(), provider, store,
        0, 196608,
    )
    assert rep.status == "ok" and len(calls) == 2 and len(rep.call_keys) == 2
    assert "SYNTHETIC_REJECTED_PAYLOAD" not in rep.text
    assert all("SYNTHETIC_REJECTED_PAYLOAD" not in r["text"] for r in provider.requests)
    assert provider.requests[0]["structured_schema"] == provider.requests[1]["structured_schema"]
    validations = [
        json.loads(p.read_text()) for p in (tmp_path / "summary_validation").glob("*.json")
    ]
    assert sorted(v["status"] for v in validations) == ["invalid_output", "ok"]


@pytest.mark.parametrize("failure", ["inline_event", "final_budget"])
def test_v7_persistent_invalid_representation_stays_null_and_never_reaches_monitor(
    tmp_path, failure,
):
    fields = draft()
    fields["important_identifiers"][0]["text"] = (
        "Independent observation E0002." if failure == "inline_event" else "x" * 2100
    )
    store = PrivateStore(tmp_path)
    provider = SyntheticProvider(store, [json.dumps(fields)] * 2)
    cfg, transcript = separate_config(), synthetic_transcript()
    rep, calls = make_representation(
        transcript, "structured_summary", cfg, provider, store, 0, 196608,
    )
    assert rep.status == ("invalid_output" if failure == "inline_event" else "budget_violation")
    assert rep.text == "" and len(calls) == 2
    decision, monitors = monitor_representation(transcript, rep, cfg, provider, 0, 196608)
    assert decision.suspicion_score is None and decision.escalate and monitors == []


def test_v7_preserves_other_prompt_contracts_and_old_v6_prompt():
    old, new = prompts(compact_config()), prompts(separate_config())
    assert old["summary_structured"] != new["summary_structured"]
    assert {k:v for k,v in old.items() if k != "summary_structured"} == {
        k:v for k,v in new.items() if k != "summary_structured"
    }
