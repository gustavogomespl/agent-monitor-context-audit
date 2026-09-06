import json
from pathlib import Path

from context_audit.dataset import parse_transcript
from context_audit.runner import build_monitor_prompt, make_representation, monitor_representation
from context_audit.runtime_models import AuditConfig
from context_audit.storage import PrivateStore


class FakeProvider:
    counter_method = "synthetic_test_character_count"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def count_text(self, model, text):
        return len(text)

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return str(len(self.calls)), next(self.responses)


def config():
    return AuditConfig(monitor_model="monitor-fixture", summarizer_model="summary-fixture")


def transcript():
    return parse_transcript(
        Path("tests/fixtures/visible_events.jsonl"),
        "t_" + "a" * 24,
        data_origin="synthetic_fixture",
    )


def reply(text, status="ok"):
    return dict(
        status=status,
        text=text,
        cost_usd=0.1,
        latency_seconds=0.2,
        usage={"input_tokens": 10, "output_tokens": 20},
        model="fixture",
    )


def test_summary_retry_is_bounded_and_counts_both_calls(tmp_path):
    provider = FakeProvider([reply("invalid without citations"), reply("Observed [E0001].")])
    rep, calls = make_representation(
        transcript(), "free_summary", config(), provider, PrivateStore(tmp_path), 0, 10000
    )
    assert rep.status == "ok" and rep.text == "Observed [E0001]."
    assert len(calls) == 2 and sum(c["cost_usd"] for c in calls) == 0.2
    assert provider.calls[0]["identity"]["attempt"] == 0
    assert provider.calls[1]["identity"]["attempt"] == 1
    assert all("label" not in c["identity"] for c in provider.calls)


def test_monitor_refusal_escalates_without_second_attempt(tmp_path):
    provider = FakeProvider([reply("", "refusal")])
    rep, _ = make_representation(
        transcript(), "full", config(), provider, PrivateStore(tmp_path), 0, 10000
    )
    result, calls = monitor_representation(transcript(), rep, config(), provider, 0, 10000)
    assert result.status == "refusal" and result.escalate and result.suspicion_score is None
    assert len(calls) == 1


def test_monitor_unknown_evidence_is_invalid_and_never_safe(tmp_path):
    text = json.dumps(
        dict(
            suspicion_score=0,
            brief_justification="okay",
            evidence_event_ids=["E9999"],
            uncertainties=[],
        )
    )
    provider = FakeProvider([reply(text), reply(text)])
    rep, _ = make_representation(
        transcript(), "full", config(), provider, PrivateStore(tmp_path), 0, 10000
    )
    result, _ = monitor_representation(transcript(), rep, config(), provider, 0, 10000)
    assert result.status == "invalid_output" and result.escalate
    assert result.suspicion_score is None


def test_monitor_does_not_receive_condition_name_or_initial_task_twice(tmp_path):
    provider = FakeProvider([])
    rep, _ = make_representation(
        transcript(), "full", config(), provider, PrivateStore(tmp_path), 0, 10000
    )
    prompt = build_monitor_prompt(transcript(), rep.text)
    assert prompt.count(transcript().initial_task) == 1
    assert "condition" not in prompt
