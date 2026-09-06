"""Time-only Qwen accounting uses synthetic HTTP responses and no GPU."""

import json
from types import SimpleNamespace

import httpx
import pytest
from test_qwen_provider import MODEL, configuration, generate, setup_provider

from context_audit.provider import BudgetLedger
from context_audit.runtime_models import AuditConfig, QwenConfig
from context_audit.storage import PrivateStore


def time_config(**changes):
    return configuration(gpu_hourly_rate_usd=None, gpu_budget_hours=12, **changes)


def time_provider(tmp_path, monkeypatch, **changes):
    return setup_provider(
        tmp_path, monkeypatch, config=time_config(),
        ledger_unit="seconds", ledger_limit=43200, **changes,
    )


def test_seconds_journal_is_durable_and_separate_from_dollars(tmp_path):
    store = PrivateStore(tmp_path)
    dollars = BudgetLedger(store, 10)
    dollars.reserve("same-id", 2)
    seconds = BudgetLedger(store, 43200, unit="seconds")
    seconds.reserve("same-id", 300)
    seconds.settle("same-id", 23)
    assert BudgetLedger(store, 43200, unit="seconds").committed == 23
    assert BudgetLedger(store, 10).committed == 2
    assert store.read_journal("budget-seconds")[0]["amount"] == 300
    with pytest.raises(ValueError, match="unit"):
        BudgetLedger(store, 1, unit="hours")


def test_time_config_does_not_require_a_fictitious_price():
    time_config().validate_live()
    with pytest.raises(ValueError, match="revision"):
        QwenConfig(gpu_budget_hours=12).validate_live()
    with pytest.raises(ValueError, match="budget|rate"):
        QwenConfig(model_revision="a" * 40).validate_live()


def test_successful_elapsed_seconds_and_unknown_usd_survive_restart(tmp_path, monkeypatch):
    clock = SimpleNamespace(value=10)

    def handler(request, _):
        if request.url.path == "/v1/chat/completions":
            clock.value += 2

    provider, store, ledger, _ = time_provider(tmp_path, monkeypatch, handler=handler)
    import context_audit.qwen_provider as module

    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock.value))
    key, record = generate(provider)
    assert record["status"] == "ok" and record["cost_usd"] is None
    assert record["accounting_unit"] == "seconds" and record["accounting_amount"] == 2
    assert record["gpu_seconds"] == record["latency_seconds"] == 2
    assert ledger.committed == 2 and key in ledger.settled
    assert store.read_journal("budget-seconds")[0]["amount"] == 30
    assert not store.read_journal("budget")
    restarted, _, restored, requests = time_provider(tmp_path, monkeypatch)
    assert generate(restarted) == (key, record)
    assert restored.committed == 2 and not requests


def test_timeout_keeps_seconds_reserved_without_generating_again(tmp_path, monkeypatch):
    def handler(request, _):
        if request.url.path == "/v1/chat/completions":
            raise httpx.ReadTimeout("independent timeout", request=request)

    provider, _, ledger, _ = time_provider(tmp_path, monkeypatch, handler=handler)
    key, record = generate(provider)
    assert record["cost_usd"] is None and record["cost_is_upper_bound"]
    assert record["gpu_seconds"] == record["accounting_amount"] == 30
    assert ledger.committed == 30 and key not in ledger.settled
    restarted, _, restored, requests = time_provider(tmp_path, monkeypatch)
    assert generate(restarted) == (key, record)
    assert restored.committed == 30 and not requests


def test_interrupted_time_settlement_recovers_without_replay(tmp_path, monkeypatch):
    provider, store, ledger, requests = time_provider(tmp_path, monkeypatch)
    settle = ledger.settle

    def fail(*_):
        raise OSError("independent write interruption")

    monkeypatch.setattr(ledger, "settle", fail)
    with pytest.raises(OSError):
        generate(provider)
    key = next(iter(ledger.amounts))
    saved = store.get("calls", key)
    monkeypatch.setattr(ledger, "settle", settle)
    assert generate(provider) == (key, saved)
    assert ledger.amounts[key] == saved["gpu_seconds"]
    assert key in ledger.settled
    assert sum(r.url.path == "/v1/chat/completions" for r, _ in requests) == 1


def test_missing_time_response_stays_reserved_and_is_not_replayed(tmp_path, monkeypatch):
    provider, store, ledger, requests = time_provider(tmp_path, monkeypatch)
    put = store.put

    def fail_response(namespace, key, value):
        if namespace == "calls":
            raise OSError("independent response write interruption")
        put(namespace, key, value)

    monkeypatch.setattr(store, "put", fail_response)
    with pytest.raises(OSError):
        generate(provider)
    key = next(iter(ledger.amounts))
    monkeypatch.setattr(store, "put", put)
    resumed, record = generate(provider)
    assert resumed == key and record["error"] == "uncertain_previous_request"
    assert record["cost_usd"] is None and record["cost_is_upper_bound"]
    assert record["accounting_amount"] == record["gpu_seconds"] == ledger.committed == 30
    assert sum(r.url.path == "/v1/chat/completions" for r, _ in requests) == 1


def test_optional_price_estimates_cost_without_becoming_a_dollar_cap(tmp_path, monkeypatch):
    provider, _, ledger, _ = setup_provider(
        tmp_path, monkeypatch, config=configuration(gpu_budget_hours=12),
        ledger_unit="seconds", ledger_limit=43200,
    )
    _, record = generate(provider)
    assert record["cost_usd"] == pytest.approx(record["gpu_seconds"] * 3.6 / 3600)
    assert ledger.committed == record["gpu_seconds"]


def test_seconds_cap_prevents_request_before_network_generation(tmp_path, monkeypatch):
    provider, _, ledger, requests = time_provider(tmp_path, monkeypatch)
    ledger.reserve("prior", 43190)
    _, record = generate(provider)
    assert record["status"] == "budget_violation" and record["error"] == "gpu_time_cap"
    assert record["cost_usd"] is None
    assert all(r.url.path != "/v1/chat/completions" for r, _ in requests)


def test_cli_allows_explicit_qwen_time_cap_without_usd(tmp_path, monkeypatch, capsys):
    import context_audit.cli as cli
    import context_audit.dataset as dataset
    import context_audit.runner as runner

    config = AuditConfig(
        provider="qwen_local", qwen=time_config(), monitor_model=MODEL, summarizer_model=MODEL,
    )
    monkeypatch.setattr(runner, "load_config", lambda _: config)
    monkeypatch.setattr(dataset, "load_dataset", lambda *a: ([], []))
    monkeypatch.setattr(dataset, "load_canaries", lambda *a: ["independent notice"])
    monkeypatch.setattr(cli, "_dataset_manifest", lambda _: {})
    invoked = []

    def run(*args, **kwargs):
        invoked.append(kwargs)
        return {"status": "executed"}

    monkeypatch.setattr(runner, "run_experiment", run)
    assert cli.main(["run", "--config", "unused", "--live"]) == 0
    assert invoked[0]["max_cost_usd"] is None
    assert json.loads(capsys.readouterr().out)["status"] == "executed"
    monkeypatch.setattr(runner, "load_config", lambda _: AuditConfig())
    assert cli.main(["run", "--config", "unused", "--live"]) == 2
    assert len(invoked) == 1


@pytest.mark.parametrize("cap", [0, -1, float("nan"), float("inf")])
def test_invalid_dollar_values_do_not_mean_unlimited(cap):
    from context_audit.runner import validate_run_budget

    cfg = AuditConfig(
        provider="qwen_local", qwen=time_config(), monitor_model=MODEL, summarizer_model=MODEL,
    )
    with pytest.raises(ValueError, match="finite"):
        validate_run_budget(cfg, cap)
