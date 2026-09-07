"""Independent synthetic checks for bounded, evidence-preserving summary regeneration."""

import json
from pathlib import Path

import pytest

from context_audit.dataset import parse_transcript
from context_audit.runner import make_representation, monitor_representation
from context_audit.runtime_models import AuditConfig
from context_audit.storage import PrivateStore


class SyntheticProvider:
    counter_method = "synthetic_character_count_not_model_tokens"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def count_text(self, model, text):
        return len(text)

    def generate(self, **request):
        self.requests.append(request)
        return f"synthetic-call-{len(self.requests)}", dict(
            status="ok", text=next(self.responses), cost_usd=0.01
        )


def synthetic_transcript():
    transcript = parse_transcript(
        Path("tests/fixtures/visible_events.jsonl"),
        "t_" + "b" * 24,
        data_origin="synthetic_fixture",
    )
    transcript.events[-1].text = "The independent toy collection has blue blocks. " * 200
    return transcript


def structured_text(*, uncited_field=None):
    fields = dict(
        environment_and_state=["Toy collection inspected [E0002]."],
        actions_and_observed_results=["Blue blocks observed [E0003]."],
        important_identifiers=["/demo/blocks.txt [E0002]."],
        contradictions_or_missing_information=["Completeness not established [E0003]."],
        source_event_ids=["E0002", "E0003"],
    )
    if uncited_field:
        fields[uncited_field] = ["SYNTHETIC_CANDIDATE_ONLY must not enter a retry."]
    return json.dumps(fields)


def run_summary(tmp_path, condition, responses):
    provider = SyntheticProvider(responses)
    config = AuditConfig(monitor_model="synthetic-monitor", summarizer_model="synthetic-summary")
    store = PrivateStore(tmp_path)
    representation, calls = make_representation(
        synthetic_transcript(), condition, config, provider, store, 0, 50000
    )
    return representation, calls, provider, store, config


@pytest.mark.parametrize("condition", ["free_summary", "structured_summary"])
def test_length_regeneration_keeps_citations_and_shared_headroom(tmp_path, condition):
    candidate = "SYNTHETIC_CANDIDATE_ONLY " + "word " * 300 + "[E0003]"
    valid = "Blue blocks observed [E0003]." if condition == "free_summary" else structured_text()
    representation, calls, provider, store, _ = run_summary(
        tmp_path, condition, [candidate, valid]
    )
    assert representation.status == "ok" and representation.text == valid
    assert representation.budget == 1024 and len(calls) == 2
    assert sum(call["cost_usd"] for call in calls) == pytest.approx(0.02)
    first, retry = provider.requests
    assert first["system"] == retry["system"]
    assert "60-80%" in first["system"] and "80–100%" not in first["system"]
    assert all("614-819" in request["text"] for request in provider.requests)
    assert "ALL constraints together" in retry["text"]
    assert "identifiers" in retry["text"] and "missing information" in retry["text"]
    assert "SYNTHETIC_CANDIDATE_ONLY" not in retry["text"]
    assert first["text"].split("<history>", 1)[1] == retry["text"].split("<history>", 1)[1]
    assert [request["identity"]["attempt"] for request in provider.requests] == [0, 1]
    diagnostics = sorted(
        (
            json.loads(path.read_text())
            for path in (store.root / "summary_validation").glob("*.json")
        ),
        key=lambda value: value["attempt"],
    )
    assert [item["reason_code"] for item in diagnostics] == ["token_ceiling_exceeded", "valid"]
    assert [item["measured_tokens"] for item in diagnostics] == [len(candidate), len(valid)]
    assert all(item["token_ceiling"] == 1024 for item in diagnostics)
    assert "SYNTHETIC_CANDIDATE_ONLY" not in json.dumps(diagnostics)


@pytest.mark.parametrize("field", [
    "environment_and_state", "actions_and_observed_results", "important_identifiers",
    "contradictions_or_missing_information",
])
def test_uncited_structured_strings_remain_failures_after_two_attempts(tmp_path, field):
    candidate = structured_text(uncited_field=field)
    representation, calls, provider, store, config = run_summary(
        tmp_path, "structured_summary", [candidate, candidate]
    )
    assert representation.status == "invalid_output" and representation.text == ""
    assert representation.measured_tokens == 0 and len(calls) == 2
    assert len(provider.requests) == 2
    assert "Each string in ALL first four fields" in provider.requests[-1]["system"]
    diagnostics = [
        json.loads(path.read_text()) for path in (store.root / "summary_validation").glob("*.json")
    ]
    assert len(diagnostics) == 2
    assert {item["reason_code"] for item in diagnostics} == {"uncited_structured_claim"}
    result, monitor_calls = monitor_representation(
        synthetic_transcript(), representation, config, provider, 0, 50000
    )
    assert monitor_calls == [] and result.suspicion_score is None and result.escalate


def test_two_oversize_outputs_are_not_truncated_or_granted_an_extra_attempt(tmp_path):
    candidate = "blue " * 300 + "[E0003]"
    representation, calls, provider, _, _ = run_summary(
        tmp_path, "free_summary", [candidate, candidate]
    )
    assert representation.status == "budget_violation"
    assert representation.text == "" and len(calls) == len(provider.requests) == 2
    assert all(request["max_tokens"] == 1600 for request in provider.requests)


def test_short_valid_summary_is_accepted_without_padding_or_regeneration(tmp_path):
    valid = "Blue blocks observed [E0003]."
    representation, calls, _, _, _ = run_summary(tmp_path, "free_summary", [valid])
    assert representation.status == "ok" and representation.text == valid
    assert representation.measured_tokens < representation.budget * 0.6 and len(calls) == 1


def test_summary_diagnostics_preserve_canaries_without_exposing_them_to_the_model(tmp_path):
    provider = SyntheticProvider(["No citations", "Blue blocks observed [E0003]."])
    provider.canaries = ["Independent synthetic private canary; never model input."]
    config = AuditConfig(monitor_model="synthetic-monitor", summarizer_model="synthetic-summary")
    store = PrivateStore(tmp_path)
    representation, _ = make_representation(
        synthetic_transcript(), "free_summary", config, provider, store, 0, 50000
    )
    assert representation.status == "ok"
    diagnostics = [
        json.loads(path.read_text()) for path in (store.root / "summary_validation").glob("*.json")
    ]
    assert len(diagnostics) == 2
    assert all(item["canaries"] == provider.canaries for item in diagnostics)
    assert provider.canaries[0] not in json.dumps(provider.requests)
    assert provider.canaries[0] not in representation.text


def test_preflight_checks_the_longer_regeneration_before_any_generation(tmp_path, monkeypatch):
    import context_audit.runner as runner

    class InventoryBoundary(SyntheticProvider):
        def model_info(self, model):
            return {"id": model, "max_input_tokens": 2400}

        def count_request(self, model, system, text):
            return 900 if "Regenerate once from the original history" in text else 100

        def generate(self, **request):
            pytest.fail("Context preflight must stop before generation")

    provider = InventoryBoundary([])
    monkeypatch.setattr(runner, "AnthropicProvider", lambda *args, **kwargs: provider)
    monkeypatch.setattr(runner, "code_state", lambda: {"commit": "a" * 40, "hash": "synthetic"})
    config = AuditConfig(
        run_dir=str(tmp_path),
        monitor_model="synthetic-monitor", summarizer_model="synthetic-summary",
    )
    with pytest.raises(ValueError, match="transcripts exceed context"):
        runner._run_locked(
            config, [synthetic_transcript()], {}, {"dataset_commit": "b" * 40},
            {"code_hash": "synthetic", "prompts": {}}, 1.0, [], {}, {},
        )
    inventory = json.loads((tmp_path / "manifests/preflight.json").read_text())
    assert inventory["items"][0]["summary_input_tokens"] == 900
    assert inventory["context_limit_ids"] == [synthetic_transcript().transcript_id]
