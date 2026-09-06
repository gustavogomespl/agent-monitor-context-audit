"""Public artifacts must contain only validated numeric rows and opaque IDs."""

import json

import pandas as pd
import pytest
from test_metrics import numeric_rows


def test_no_data_creates_honest_report_without_figures(tmp_path):
    from context_audit.reporting import generate_report

    result = generate_report(tmp_path / "absent.csv", tmp_path / "results")
    assert result["status"] == "experiment_not_executed"
    assert "Experiment not executed" in (tmp_path / "results" / "findings.md").read_text()
    assert not list((tmp_path / "results").rglob("*.png"))
    assert json.loads((tmp_path / "results" / "metrics.json").read_text())["primary"] is None


def test_empirical_report_rejects_fixture_origin_and_text_columns(tmp_path):
    from context_audit.reporting import generate_report

    source = tmp_path / "scores.csv"
    pd.DataFrame(numeric_rows()).to_csv(source, index=False)
    with pytest.raises(ValueError, match="synthetic"):
        generate_report(source, tmp_path / "results", n_bootstrap=20)
    rows = numeric_rows()
    for row in rows:
        row["data_origin"] = "sleight_bench"
        row["brief_justification"] = "Private source content must not be public"
    pd.DataFrame(rows).to_csv(source, index=False)
    with pytest.raises(ValueError, match="column"):
        generate_report(source, tmp_path / "results", n_bootstrap=20)


def test_qualitative_selection_is_deterministic_bounded_and_opaque():
    from context_audit.reporting import select_qualitative_cases

    rows = numeric_rows(12)
    selected = select_qualitative_cases(rows, seed=7)
    assert selected == select_qualitative_cases(list(reversed(rows)), seed=7)
    assert len(selected) <= 12
    assert sum(row["selection_group"] == "discordance" for row in selected) <= 6
    assert sum(row["selection_group"] == "control" for row in selected) <= 6
    assert all("transcript_id" in row and not row["review_category"] for row in selected)


def test_reports_produce_all_figures_and_explicit_denominators(tmp_path):
    from context_audit.reporting import generate_report

    # Artificial independent numeric records exercise the real-data gate in isolation.
    # They live only in pytest's temporary directory and are never empirical artifacts.
    rows = numeric_rows()
    for row in rows:
        row["data_origin"] = "sleight_bench"
    source = tmp_path / "scores.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    output = tmp_path / "results"
    result = generate_report(source, output, n_bootstrap=25)
    assert result["primary"]["n_transcripts"] == 4
    assert {p.name for p in (output / "figures").glob("*.png")} == {
        "auroc.png",
        "threshold_rates.png",
        "cost_performance.png",
    }
    report = (output / "findings.md").read_text()
    assert "didactic-model" in report
    assert "4/4" in report
    assert "0.5" in report


def test_manifest_verifies_planned_coverage_and_exposes_missing_entire_pair(tmp_path):
    from context_audit.reporting import generate_report

    rows = numeric_rows(3)
    for row in rows:
        row["data_origin"] = "sleight_bench"
    labels = {
        row["transcript_id"]: {
            key: row[key] for key in ("transcript_id", "scenario_id", "family_id", "split", "label")
        }
        for row in rows
    }
    manifest = {
        "data_origin": "sleight_bench",
        "planned_calls": [
            {key: row[key] for key in ("transcript_id", "condition", "repetition")} for row in rows
        ],
        "evaluation_labels": list(labels.values()),
    }
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps(manifest))
    source = tmp_path / "scores.csv"
    pd.DataFrame(rows[:16]).to_csv(source, index=False)
    result = generate_report(source, tmp_path / "results", n_bootstrap=10, manifest_path=path)
    assert result["primary"]["planned_coverage_verified"]
    assert result["primary"]["coverage"] == pytest.approx(2 / 3)
    assert result["conditions"]["full"]["n_missing"] == 2
    pd.DataFrame([rows[4]]).to_csv(source, index=False)
    interrupted = generate_report(source, tmp_path / "results", n_bootstrap=10, manifest_path=path)
    assert interrupted["primary"]["n_observations"] == 0
    assert interrupted["n_observations"] == 1
    manifest["planned_calls"].pop()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="four"):
        generate_report(source, tmp_path / "results", n_bootstrap=10, manifest_path=path)
