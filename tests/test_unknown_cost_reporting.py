"""Unknown GPU prices stay unknown throughout synthetic numeric reporting."""

import io
import json

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from test_metrics import numeric_rows

COST_COLUMNS = (
    "summary_cost_usd", "monitor_cost_usd", "infrastructure_cost_usd", "cost_usd"
)


def unpriced_rows():
    rows = numeric_rows()
    for row in rows:
        row.update(dict.fromkeys(COST_COLUMNS))
    return rows


def test_unknown_usd_survives_numeric_csv_round_trip_and_analysis():
    from context_audit.metrics import report_metrics, validate_rows

    frame = validate_rows(unpriced_rows())
    restored = validate_rows(pd.read_csv(io.StringIO(frame.to_csv(index=False))))
    assert restored[list(COST_COLUMNS)].isna().all().all()
    metrics = report_metrics(restored, n_bootstrap=5)
    assert metrics["primary"]["delta_auroc"]["estimate"] == 0.5
    assert metrics["costs"]["recorded_generation_cost_usd"] is None
    assert metrics["costs"]["upper_bound_cost_usd"] is None
    assert metrics["costs"]["n_cost_unknown"] == len(frame)
    assert not metrics["costs"]["monetary_cost_available"]
    for group in metrics["conditions"].values():
        assert all(group[column] is None for column in COST_COLUMNS)
        assert group["n_cost_unknown"] == 4
        assert group["latency_seconds"]["mean"] == 1.5
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize("column", COST_COLUMNS)
def test_partially_known_row_costs_are_rejected(column):
    from context_audit.metrics import validate_rows

    rows = unpriced_rows()
    rows[0][column] = 0
    with pytest.raises(ValueError, match="all known or all unknown"):
        validate_rows(rows)


def test_legacy_missing_infrastructure_cannot_make_unknown_cost_free():
    from context_audit.metrics import validate_rows

    rows = unpriced_rows()
    for row in rows:
        del row["infrastructure_cost_usd"]
    frame = validate_rows(rows)
    assert frame[list(COST_COLUMNS)].isna().all().all()
    assert validate_rows(numeric_rows())["infrastructure_cost_usd"].eq(0).all()


def test_unknown_row_prevents_partial_known_sum_being_reported_as_total():
    from context_audit.metrics import report_metrics

    rows = numeric_rows()
    for row in rows:
        row["infrastructure_cost_usd"] = 0
    rows[0].update(dict.fromkeys(COST_COLUMNS))
    metrics = report_metrics(rows, n_bootstrap=5)
    assert metrics["costs"]["recorded_generation_cost_usd"] is None
    assert metrics["conditions"]["full"]["cost_usd"] is None
    assert metrics["conditions"]["head_tail"]["cost_usd"] == pytest.approx(0.12)


@pytest.mark.parametrize("column", ["latency_seconds", "input_tokens", "cost_usd"])
def test_unknown_price_does_not_allow_other_invalid_measurements(column):
    from context_audit.metrics import validate_rows

    rows = unpriced_rows()
    rows[0][column] = float("inf")
    with pytest.raises(ValueError):
        validate_rows(rows)


def test_unpriced_report_keeps_metrics_and_figures_without_zero_cost_claim(tmp_path):
    from context_audit.reporting import generate_report

    # Independent synthetic records exercise the reporting gate only in pytest's temp dir.
    rows = unpriced_rows()
    for row in rows:
        row["data_origin"] = "sleight_bench"
    source = tmp_path / "scores.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    output = tmp_path / "results"
    metrics = generate_report(source, output, n_bootstrap=5)
    assert metrics["costs"]["recorded_generation_cost_usd"] is None
    rendered = (output / "findings.md").read_text()
    assert "Recorded execution cost: unavailable" in rendered
    assert "$0.000000" not in rendered
    assert "no GPU hourly rate was supplied" in rendered
    assert len(list((output / "figures").glob("*.png"))) == 3
    stored = json.loads((output / "metrics.json").read_text())
    assert stored["costs"]["recorded_generation_cost_usd"] is None


def test_unknown_cost_figure_has_annotation_and_no_free_points(tmp_path, monkeypatch):
    from context_audit import reporting
    from context_audit.metrics import report_metrics

    figures = {}

    def capture(fig, path):
        figures[path.name] = fig

    monkeypatch.setattr(reporting, "_save", capture)
    try:
        reporting._figures(report_metrics(unpriced_rows(), n_bootstrap=5), tmp_path)
        ax = figures["cost_performance.png"].axes[0]
        assert len(ax.collections) == 0
        assert any("USD cost unavailable" in label.get_text() for label in ax.texts)
    finally:
        for fig in figures.values():
            plt.close(fig)


def test_export_preserves_unknown_costs_in_sanitized_numeric_csv(tmp_path, monkeypatch):
    from context_audit import cli, dataset, runner
    from context_audit.runtime_models import AuditConfig
    from context_audit.storage import digest

    rows = unpriced_rows()
    for row in rows:
        row["data_origin"] = "sleight_bench"
    run = tmp_path / "private-run"
    (run / "manifests").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(run / "scores.csv", index=False)
    manifest = {
        key: {} for key in (
            "dataset_commit", "code_commit", "code_hash", "seeds", "prompt_hashes",
            "models", "failure_policy", "created_at", "prices", "counter_method",
        )
    }
    dataset_manifest = {"normalized_content_digest": "independent-synthetic-fixture"}
    manifest.update(
        run_id="synthetic-export-contract", data_origin="sleight_bench", concurrency=1,
        planned_transcripts=sorted({row["transcript_id"] for row in rows}),
        planned_calls=[
            {key: row[key] for key in ("transcript_id", "condition", "repetition")}
            for row in rows
        ],
        config=AuditConfig().model_dump(), dataset_manifest_hash=digest(dataset_manifest),
    )
    (run / "manifests/run.json").write_text(json.dumps(manifest))
    # Replace only private-data acquisition boundaries, retaining export validation.
    monkeypatch.setattr(runner, "require_private_path", lambda path, root: None)
    monkeypatch.setattr(cli, "_dataset_manifest", lambda config: dataset_manifest)
    monkeypatch.setattr(dataset, "load_dataset", lambda path, split: ([], []))
    output = tmp_path / "public"
    result = cli.export_results(run, output)
    assert result["rows"] == len(rows)
    assert pd.read_csv(output / "public_scores.csv")[list(COST_COLUMNS)].isna().all().all()
