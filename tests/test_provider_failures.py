"""Count-endpoint failures must remain explicit and retain prior generation costs."""

from pathlib import Path
from types import SimpleNamespace

import anthropic
import pytest

from context_audit.dataset import parse_transcript
from context_audit.provider import AnthropicProvider, BudgetLedger, Price
from context_audit.runner import make_representation, monitor_representation
from context_audit.runtime_models import AuditConfig
from context_audit.storage import PrivateStore


def test_count_connection_error_is_a_persisted_nonbillable_api_failure(tmp_path, monkeypatch):
    generated = []

    def fail_count(**kwargs):
        raise anthropic.APIConnectionError(request=None)

    def forbidden_generation(**kwargs):
        generated.append(kwargs)
        pytest.fail("Token-count failure must stop generation before reservation")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "independent-fixture-key")
    monkeypatch.setattr(
        anthropic,
        "Anthropic",
        lambda **kwargs: SimpleNamespace(
            messages=SimpleNamespace(count_tokens=fail_count, create=forbidden_generation)
        ),
    )
    store = PrivateStore(tmp_path)
    ledger = BudgetLedger(store, 1)
    price = Price(
        input_per_million=1,
        output_per_million=1,
        cache_read_per_million=0,
        cache_write_per_million=0,
    )
    provider = AnthropicProvider(store, ledger, {"fixture-model": price})
    key, result = provider.generate(
        model="fixture-model",
        system="Independent instruction",
        text="Independent data",
        max_tokens=100,
        context_window=1000,
        identity={"attempt": 0},
    )
    assert result["status"] == "api_error" and result["error"] == "token_count_failed"
    assert result["cost_usd"] == 0 and ledger.committed == 0 and generated == []
    assert store.get("calls", key)["status"] == "api_error"


@pytest.mark.parametrize("condition", ["free_summary", "head_tail"])
def test_count_failure_preserves_realized_cost_and_escalates(tmp_path, condition):
    from context_audit.provider import TokenCountError

    source = parse_transcript(
        Path("tests/fixtures/visible_events.jsonl"),
        "t_" + "a" * 24,
        data_origin="synthetic_fixture",
    )
    summary = "Observed result [E0003]."
    produced = []

    class EndpointBoundary:
        counter_method = "independent_fixture_counter"

        def count_text(self, model, text):
            if text == summary or "content omitted" in text:
                raise TokenCountError("Independent count-endpoint failure")
            return len(text)

        def generate(self, **kwargs):
            produced.append(kwargs)
            return "call-one", dict(
                status="ok",
                text=summary,
                cost_usd=0.03,
                usage={"input_tokens": 10, "output_tokens": 20},
                latency_seconds=0.1,
                model="fixture-model",
            )

    provider = EndpointBoundary()
    config = AuditConfig(monitor_model="fixture-model", summarizer_model="fixture-model")
    representation, summary_calls = make_representation(
        source, condition, config, provider, PrivateStore(tmp_path), 0, 10000
    )
    result, monitor_calls = monitor_representation(
        source, representation, config, provider, 0, 10000
    )
    assert representation.status == result.status == "api_error"
    assert result.suspicion_score is None and result.escalate
    assert representation.text == "" and representation.measured_tokens == 0
    assert representation.body_tokens > 0 and monitor_calls == []
    if condition == "free_summary":
        assert representation.call_keys == ["call-one"]
        assert len(produced) == 1 and sum(c["cost_usd"] for c in summary_calls) == 0.03
    else:
        assert produced == [] and summary_calls == []
