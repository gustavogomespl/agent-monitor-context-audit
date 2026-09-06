"""Explicit external allocation debits use independent synthetic ledgers."""

import pytest

from context_audit.provider import BudgetLedger
from context_audit.storage import PrivateStore


def test_external_allocation_debit_is_idempotent_and_shared_across_phases(tmp_path):
    from context_audit.colab import initialize_gpu_budget, record_external_gpu_time

    initialize_gpu_budget(tmp_path / "qwen-pilot", 12, previously_used_seconds=600)
    first = record_external_gpu_time(
        tmp_path / "qwen-pilot", 12,
        usage_id="resume-20260906-a", elapsed_seconds=900, confirmed=True,
    )
    again = record_external_gpu_time(
        tmp_path / "qwen-development", 12,
        usage_id="resume-20260906-a", elapsed_seconds=900, confirmed=True,
    )
    assert first == again
    assert again["committed_seconds"] == 1500
    assert again["remaining_seconds"] == 41700
    ledger = BudgetLedger(PrivateStore(tmp_path / "gpu_budget"), 43200, unit="seconds")
    assert ledger.amounts["external-resume-20260906-a"] == 900
    assert "external-resume-20260906-a" in ledger.settled
    with pytest.raises(ValueError, match="differs|changed"):
        record_external_gpu_time(
            tmp_path / "qwen-pilot", 12,
            usage_id="resume-20260906-a", elapsed_seconds=100, confirmed=True,
        )
    assert BudgetLedger(ledger.store, 43200, unit="seconds").committed == 1500


@pytest.mark.parametrize("changes", [
    {"confirmed": False}, {"elapsed_seconds": -1}, {"elapsed_seconds": float("inf")},
    {"usage_id": "../unsafe"}, {"usage_id": ""}, {"max_gpu_hours": 0},
])
def test_invalid_external_debit_never_creates_accounting(tmp_path, changes):
    from context_audit.colab import record_external_gpu_time

    kwargs = dict(
        max_gpu_hours=12, usage_id="synthetic-resume", elapsed_seconds=900, confirmed=True,
    )
    kwargs.update(changes)
    with pytest.raises(ValueError):
        record_external_gpu_time(tmp_path / "qwen-pilot", **kwargs)
    assert not (tmp_path / "gpu_budget").exists()


def test_external_overrun_preserves_actual_use_then_stops_future_budget_access(tmp_path):
    from context_audit.colab import initialize_gpu_budget, record_external_gpu_time

    run = tmp_path / "qwen-pilot"
    initialize_gpu_budget(run, 12, previously_used_seconds=43100)
    with pytest.raises(ValueError, match="exceeded|exceed"):
        record_external_gpu_time(
            run, 12, usage_id="resume-overrun", elapsed_seconds=200, confirmed=True,
        )
    store = PrivateStore(tmp_path / "gpu_budget")
    journal = store.read_journal("budget-seconds")
    assert journal[-1]["action"] == "settle"
    assert journal[-1]["call_id"] == "external-resume-overrun"
    assert journal[-1]["amount"] == 200
    assert store.get("external_usage", "external-resume-overrun")["elapsed_seconds"] == 200
    with pytest.raises(ValueError, match="exceed"):
        initialize_gpu_budget(run, 12, previously_used_seconds=43100)


def test_external_debit_recovers_interrupted_settlement_without_charging_twice(tmp_path):
    from context_audit.colab import initialize_gpu_budget, record_external_gpu_time

    run = tmp_path / "qwen-pilot"
    initialize_gpu_budget(run, 12)
    store = PrivateStore(tmp_path / "gpu_budget")
    store.put("external_usage", "external-resume-a", {"elapsed_seconds": 900})
    BudgetLedger(store, 43200, unit="seconds").reserve("external-resume-a", 900)
    receipt = record_external_gpu_time(
        run, 12, usage_id="resume-a", elapsed_seconds=900, confirmed=True,
    )
    assert receipt["committed_seconds"] == 900
    assert receipt["uncertain_sessions"] == []
    assert len(store.read_journal("budget-seconds")) == 2


def test_external_usage_cannot_change_the_pinned_ceiling(tmp_path):
    from context_audit.colab import initialize_gpu_budget, record_external_gpu_time

    run = tmp_path / "qwen-pilot"
    initialize_gpu_budget(run, 12)
    with pytest.raises(ValueError, match="differs"):
        record_external_gpu_time(
            run, 24, usage_id="resume-a", elapsed_seconds=900, confirmed=True,
        )
    assert initialize_gpu_budget(run, 12)["committed_seconds"] == 0


def test_interrupted_dollar_session_reports_conservative_seconds(tmp_path):
    from context_audit.colab import finalize_gpu_costs

    store = PrivateStore(tmp_path / "gpu_sessions")
    ledger = BudgetLedger(store, 1)
    ledger.reserve("synthetic-lost-session", 0.06)
    store.put("sessions", "synthetic-lost-session", {"hourly_rate_usd": 3.6})
    receipt = finalize_gpu_costs(tmp_path, 1)
    assert receipt["managed_session_seconds"] == pytest.approx(60)
    assert receipt["time_is_upper_bound"]


def test_dollar_session_missing_time_metadata_is_not_reported_as_zero_seconds(tmp_path):
    from context_audit.colab import finalize_gpu_costs

    store = PrivateStore(tmp_path / "gpu_sessions")
    ledger = BudgetLedger(store, 1)
    ledger.reserve("synthetic-lost-session", 0.06)
    receipt = finalize_gpu_costs(tmp_path, 1)
    assert receipt["managed_session_seconds"] is None
