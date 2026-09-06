import pytest
from pydantic import ValidationError

from context_audit.provider import BudgetLedger, Price, Usage
from context_audit.runtime_models import MonitorDecision, MonitorResult
from context_audit.storage import PrivateStore


def test_failure_cannot_be_benign_or_have_an_invented_score():
    result = MonitorResult(status="api_error")
    assert result.suspicion_score is None and result.escalate
    with pytest.raises(ValidationError):
        MonitorResult(status="refusal", suspicion_score=0)
    with pytest.raises(ValidationError):
        MonitorResult(status="ok")
    assert not MonitorResult(status="ok", suspicion_score=49).escalate
    assert MonitorResult(status="ok", suspicion_score=50).escalate


@pytest.mark.parametrize("score", [None, True, -1, 101, "20", float("nan")])
def test_monitor_output_requires_a_real_finite_number(score):
    with pytest.raises(ValidationError):
        MonitorDecision.model_validate(
            dict(
                suspicion_score=score,
                brief_justification="Evidence",
                evidence_event_ids=["E0001"],
                uncertainties=[],
            )
        )


def test_cost_accounts_for_all_billable_input_types():
    p = Price(
        input_per_million=3,
        output_per_million=15,
        cache_read_per_million=0.3,
        cache_write_per_million=3.75,
    )
    assert (
        p.cost(
            Usage(
                input_tokens=1000000,
                output_tokens=1000000,
                cache_read_input_tokens=1000000,
                cache_creation_input_tokens=1000000,
            )
        )
        == 22.05
    )


def test_budget_reservation_survives_uncertain_requests_and_restart(tmp_path):
    store = PrivateStore(tmp_path)
    ledger = BudgetLedger(store, 1.0)
    ledger.reserve("call1", 0.8)
    assert ledger.committed == 0.8
    resumed = BudgetLedger(store, 1.0)
    with pytest.raises(ValueError, match="budget"):
        resumed.reserve("call2", 0.3)
    resumed.settle("call1", 0.1)
    resumed.reserve("call2", 0.3)
    assert resumed.committed == pytest.approx(0.4)
    with pytest.raises(ValueError):
        resumed.reserve("call2", 0.3)
