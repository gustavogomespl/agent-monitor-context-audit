"""Independent character-count fixtures exercise the versioned compressed ceiling."""

import json

import pytest
from test_colab_experiment_version import bootstrap, write_json
from test_structured_summary_v3 import SyntheticProvider, draft, synthetic_transcript

from context_audit.render import render_body
from context_audit.runner import make_representation, monitor_representation
from context_audit.storage import PrivateStore
from context_audit.structured_summary import assemble_structured_summary


def configured(tmp_path, version):
    write_json(tmp_path / "configuration/model-pin.json", {
        "model_id": "Qwen/Qwen3.8-27B", "model_revision": "b" * 40,
        "vllm_version": "0.28.0", "runtime_versions": {},
    })
    namespace = bootstrap(tmp_path, EXPERIMENT_VERSION=version)
    namespace["prepare_version_workspace"]()
    return namespace["configured_phase"]("pilot")


def candidate(condition, *, oversized=False):
    text = "Independent blue blocks counted. " * (75 if oversized else 35)
    if condition == "free_summary":
        return text + " [E0003]"
    fields = draft()
    fields["actions_and_observed_results"][0]["text"] = text
    return json.dumps(fields)


@pytest.mark.parametrize("condition", ["head_tail", "free_summary", "structured_summary"])
def test_new_ceiling_reaches_every_compressed_condition_without_cutting_valid_summaries(
    tmp_path, condition,
):
    cfg = configured(tmp_path, "summary-v4")
    store = PrivateStore(tmp_path / "synthetic-run")
    raw = candidate(condition)
    provider = SyntheticProvider(store, [raw])
    transcript = synthetic_transcript()
    representation, calls = make_representation(
        transcript, condition, cfg, provider, store, 0, 196608,
    )
    assert representation.budget == 2048
    assert representation.status == "ok"
    assert 1024 < representation.measured_tokens <= 2048
    if condition == "head_tail":
        assert not calls and not provider.requests
    else:
        expected = raw if condition == "free_summary" else assemble_structured_summary(
            raw, {event.event_id for event in transcript.events}
        )
        assert representation.text == expected
        assert len(calls) == 1 and provider.requests[0]["max_tokens"] == 3200
        assert "1228-1638" in provider.requests[0]["text"]


def test_full_history_stays_integral_under_the_new_ceiling(tmp_path):
    cfg = configured(tmp_path, "summary-v4")
    store = PrivateStore(tmp_path / "synthetic-run")
    provider = SyntheticProvider(store, [])
    transcript = synthetic_transcript()
    representation, calls = make_representation(
        transcript, "full", cfg, provider, store, 0, 196608,
    )
    assert representation.text == render_body(transcript)
    assert representation.budget == representation.measured_tokens > 2048
    assert not calls and not provider.requests


@pytest.mark.parametrize("condition", ["free_summary", "structured_summary"])
def test_new_ceiling_keeps_two_attempts_null_failures_and_no_silent_truncation(tmp_path, condition):
    cfg = configured(tmp_path, "summary-v4")
    raw = candidate(condition, oversized=True)
    store = PrivateStore(tmp_path / "synthetic-run")
    provider = SyntheticProvider(store, [raw, raw])
    transcript = synthetic_transcript()
    representation, calls = make_representation(
        transcript, condition, cfg, provider, store, 0, 196608,
    )
    assert representation.budget == 2048 and representation.status == "budget_violation"
    assert representation.text == "" and representation.measured_tokens == 0
    assert len(calls) == len(provider.requests) == 2
    assert all(request["max_tokens"] == 3200 for request in provider.requests)
    monitor, monitor_calls = monitor_representation(
        transcript, representation, cfg, provider, 0, 196608,
    )
    assert monitor.suspicion_score is None and monitor.escalate and not monitor_calls


def test_prior_version_retains_old_ceiling_for_the_same_independent_candidate(tmp_path):
    cfg = configured(tmp_path, "summary-v3")
    raw = candidate("structured_summary")
    store = PrivateStore(tmp_path / "synthetic-run")
    provider = SyntheticProvider(store, [raw, raw])
    representation, calls = make_representation(
        synthetic_transcript(), "structured_summary", cfg, provider, store, 0, 196608,
    )
    assert representation.budget == 1024 and representation.status == "budget_violation"
    assert len(calls) == 2
    assert all(request["max_tokens"] == 1600 for request in provider.requests)
