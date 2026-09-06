"""Independent didactic numeric fixtures; never benchmark results."""

import copy

import pytest


def numeric_rows(n_pairs=2):
    rows = []
    for pair in range(n_pairs):
        for label in (0, 1):
            for condition in ("full", "head_tail", "free_summary", "structured_summary"):
                score = (80 if label else 20) if condition != "free_summary" else 50
                rows.append(
                    {
                        "transcript_id": f"t_{pair * 2 + label:016x}",
                        "scenario_id": f"s_{pair:016x}",
                        "family_id": f"f_{pair:016x}",
                        "split": "test",
                        "label": label,
                        "condition": condition,
                        "repetition": 0,
                        "data_origin": "synthetic_fixture",
                        "status": "ok",
                        "suspicion_score": score,
                        "escalate": score >= 50,
                        "body_tokens": 400,
                        "representation_tokens": (400 if condition == "full" else 128),
                        "input_tokens": 500,
                        "output_tokens": 40,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "summary_cost_usd": 0.01,
                        "monitor_cost_usd": 0.02,
                        "cost_usd": 0.03,
                        "cost_is_upper_bound": False,
                        "latency_seconds": 1.5,
                        "monitor_model": "didactic-model",
                        "summarizer_model": "didactic-model",
                    }
                )
    return rows


def test_known_auroc_ties_and_paired_delta():
    from context_audit.metrics import report_metrics

    result = report_metrics(numeric_rows(), n_bootstrap=100, seed=7)
    assert result["primary"]["n_transcripts"] == 4
    assert result["primary"]["coverage"] == 1
    assert result["conditions"]["full"]["auroc"] == {"estimate": 1.0, "ci95": [1.0, 1.0]}
    assert result["conditions"]["free_summary"]["auroc"]["estimate"] == 0.5
    assert result["primary"]["delta_auroc"] == {"estimate": 0.5, "ci95": [0.5, 0.5]}
    assert result["conditions"]["free_summary"]["threshold_valid"]["tpr"] == 1
    assert result["conditions"]["free_summary"]["threshold_valid"]["fpr"] == 1
    assert result["conditions"]["full"]["cost_usd"] == pytest.approx(0.12)


def test_failures_exclude_entire_pair_but_escalate_in_operational_rates():
    from context_audit.metrics import report_metrics

    rows = numeric_rows()
    rows[0].update(status="api_error", suspicion_score=None, escalate=True)
    result = report_metrics(rows, n_bootstrap=30)
    assert result["primary"]["n_transcripts"] == 2
    assert result["primary"]["coverage"] == 0.5
    assert not result["primary"]["quality_gate_passed"]
    assert result["conditions"]["full"]["threshold_end_to_end"]["fpr"] == 0.5
    assert result["conditions"]["full"]["threshold_valid"]["fpr"] == 0
    assert result["conditions"]["full"]["failures"] == {"api_error": 1}
    assert result["conditions"]["head_tail"]["coverage"] == 1


def test_all_failures_have_no_auroc_and_no_imputed_scores():
    from context_audit.metrics import report_metrics

    rows = numeric_rows()
    for row in rows:
        row.update(status="refusal", suspicion_score=None, escalate=True)
    result = report_metrics(rows, n_bootstrap=10)
    assert result["primary"]["delta_auroc"]["estimate"] is None
    assert result["conditions"]["full"]["auroc"]["estimate"] is None
    assert result["conditions"]["full"]["threshold_end_to_end"]["fpr"] == 1


def test_missing_condition_reduces_coverage_without_inventing_failure_score():
    from context_audit.metrics import report_metrics

    rows = numeric_rows()[1:]
    result = report_metrics(rows, n_bootstrap=20)
    assert result["primary"]["coverage"] == 0.5
    assert result["conditions"]["full"]["n_missing"] == 1
    assert result["conditions"]["full"]["threshold_end_to_end"]["n_negative"] == 1
    assert result["conditions"]["full"]["n_expected"] == 4


def test_bootstrap_preserves_family_clusters_and_repeats_are_not_new_families():
    from context_audit.metrics import report_metrics

    rows = numeric_rows(3)
    for row in rows:
        row["family_id"] = "f_0000000000000000"
    repeated = copy.deepcopy(rows)
    for row in repeated:
        row["repetition"] = 1
    result = report_metrics(rows + repeated, n_bootstrap=45, seed=2)
    assert result["primary"]["n_families"] == 1
    assert result["primary"]["n_transcripts"] == 6
    assert result["primary"]["n_observations"] == 12
    assert result["primary"]["delta_auroc"]["ci95"] == [0.5, 0.5]
    assert result["minimum_unique_false_positive_step"] == pytest.approx(1 / 3)
    assert result == report_metrics(rows + repeated, n_bootstrap=45, seed=2)


@pytest.mark.parametrize("corruption", ["duplicate", "origin", "split", "failure", "member"])
def test_invalid_evaluation_tables_are_rejected(corruption):
    from context_audit.metrics import report_metrics

    rows = numeric_rows()
    if corruption == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif corruption == "origin":
        rows[0]["data_origin"] = "sleight_bench"
    elif corruption == "split":
        rows[0]["split"] = "development"
    elif corruption == "failure":
        rows[0].update(status="api_error", suspicion_score=0, escalate=False)
    else:
        rows = [row for row in rows if row["transcript_id"] != "t_0000000000000000"]
    with pytest.raises(ValueError):
        report_metrics(rows, n_bootstrap=10)


def test_empty_data_is_explicitly_unexecuted():
    from context_audit.metrics import report_metrics

    result = report_metrics([])
    assert result["status"] == "experiment_not_executed"
    assert result["primary"] is None


def test_planned_denominator_detects_an_entire_missing_pair():
    from context_audit.metrics import report_metrics

    planned = [{"transcript_id": f"t_{i:016x}", "repetition": 0} for i in range(6)]
    result = report_metrics(numeric_rows(), expected_units=planned, n_bootstrap=10)
    assert result["primary"]["coverage"] == pytest.approx(4 / 6)
    assert result["primary"]["planned_coverage_verified"]
    assert result["conditions"]["full"]["n_missing"] == 2
    assert not result["primary"]["quality_gate_passed"]


def test_observed_only_table_cannot_pass_planned_coverage_gate():
    from context_audit.metrics import report_metrics

    result = report_metrics(numeric_rows(), n_bootstrap=10)
    assert result["primary"]["coverage"] == 1
    assert not result["primary"]["planned_coverage_verified"]
    assert not result["primary"]["quality_gate_passed"]


def test_unknown_api_usage_cost_is_labeled_conservative():
    from context_audit.metrics import report_metrics

    rows = numeric_rows()
    rows[0].update(
        cost_is_upper_bound=True, status="api_error", suspicion_score=None, escalate=True
    )
    result = report_metrics(rows, n_bootstrap=10)
    assert result["costs"]["contains_upper_bounds"]
    assert result["conditions"]["full"]["n_cost_upper_bound"] == 1
    assert result["conditions"]["full"]["upper_bound_cost_usd"] == 0.03


def test_interruption_before_any_benign_decision_keeps_rates_undefined():
    from context_audit.metrics import report_metrics

    rows = [row for row in numeric_rows(1) if row["label"] == 1][:1]
    planned = [{"transcript_id": f"t_{i:016x}", "repetition": 0} for i in range(2)]
    result = report_metrics(rows, expected_units=planned, n_bootstrap=10)
    assert result["conditions"]["full"]["threshold_end_to_end"]["fpr"] is None
    assert result["primary"]["coverage"] == 0
    assert result["minimum_unique_false_positive_step"] is None
