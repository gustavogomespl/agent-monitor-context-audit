"""Cumulative GPU-hour accounting uses synthetic journals and no GPU processes."""

import pytest

from context_audit.provider import BudgetLedger
from context_audit.storage import PrivateStore


def test_initial_time_debit_is_shared_and_idempotent_across_phases(tmp_path):
    from context_audit.colab import initialize_gpu_budget

    pilot = tmp_path / 'qwen-pilot'
    development = tmp_path / 'qwen-development'
    first = initialize_gpu_budget(pilot, 12, previously_used_seconds=1800)
    again = initialize_gpu_budget(development, 12, previously_used_seconds=1800)
    assert first['remaining_seconds'] == again['remaining_seconds'] == 41400
    assert first['budget_dir'] == again['budget_dir']
    ledger = BudgetLedger(PrivateStore(tmp_path / 'gpu_budget'), 43200, unit='seconds')
    assert ledger.committed == 1800
    assert len(ledger.amounts) == 1
    with pytest.raises(ValueError, match='already|changed|differs'):
        initialize_gpu_budget(pilot, 24, previously_used_seconds=1800)


def test_resume_and_next_phase_cannot_reset_time_budget(tmp_path):
    from context_audit.colab import initialize_gpu_budget

    initialize_gpu_budget(tmp_path / 'qwen-pilot', 12)
    ledger = BudgetLedger(PrivateStore(tmp_path / 'gpu_budget'), 43200, unit='seconds')
    ledger.reserve('synthetic-pilot', 3600)
    ledger.settle('synthetic-pilot', 600)
    ledger.reserve('synthetic-lost-session', 3600)
    receipt = initialize_gpu_budget(tmp_path / 'qwen-test', 12)
    assert receipt['committed_seconds'] == 4200
    assert receipt['remaining_seconds'] == 39000
    assert receipt['uncertain_sessions'] == ['synthetic-lost-session']


@pytest.mark.parametrize('hours,prior', [(0, 0), (float('inf'), 0), (12, -1), (12, 43201)])
def test_invalid_budget_or_prior_time_never_creates_a_ledger(tmp_path, hours, prior):
    from context_audit.colab import initialize_gpu_budget

    with pytest.raises(ValueError):
        initialize_gpu_budget(tmp_path / 'qwen-pilot', hours, previously_used_seconds=prior)
    assert not (tmp_path / 'gpu_budget').exists()


def test_reconcile_unknown_usd_preserves_cumulative_time(tmp_path):
    from context_audit.colab import initialize_gpu_budget, reconcile_gpu_session

    run_dir = tmp_path / 'qwen-pilot'
    initialize_gpu_budget(run_dir, 12)
    budget_dir = tmp_path / 'gpu_budget'
    ledger = BudgetLedger(PrivateStore(budget_dir), 43200, unit='seconds')
    ledger.reserve('synthetic-interrupted', 3600)
    PrivateStore(run_dir / 'gpu_sessions').put('sessions', 'synthetic-interrupted', {
        'hourly_rate_usd': None, 'gpu_budget_hours': 12,
        'gpu_budget_dir': str(budget_dir), 'deadline_epoch': 1,
    })
    reconcile_gpu_session(run_dir, 'synthetic-interrupted', elapsed_seconds=900,
                          max_cost_usd=None, confirmed=True)
    receipt = initialize_gpu_budget(tmp_path / 'qwen-development', 12)
    assert receipt['committed_seconds'] == 900
    costs = PrivateStore(run_dir).get('manifests', 'gpu_costs')
    assert costs['managed_session_cost_usd'] is None
    assert costs['managed_session_seconds'] == 900
    assert costs['remaining_gpu_seconds'] == 42300
