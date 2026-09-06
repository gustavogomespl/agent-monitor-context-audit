"""Independent regressions for private output and frozen offline analysis."""

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from test_metrics import numeric_rows

from context_audit.dataset import DatasetError, _private_dir
from context_audit.reporting import generate_report


@pytest.fixture
def private_repository(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    (root / ".gitignore").write_text("data/private\n")
    monkeypatch.chdir(root)
    return root


def test_named_private_directory_outside_approved_root_is_rejected(private_repository):
    destination = private_repository / "exports/private"
    with pytest.raises(DatasetError, match="private"):
        _private_dir(destination)
    assert not destination.exists()


def test_private_destination_requires_a_git_ignore_rule(private_repository):
    (private_repository / ".gitignore").write_text("")
    destination = private_repository / "data/private"
    with pytest.raises(DatasetError, match="ignor"):
        _private_dir(destination)
    assert not destination.exists()


def test_private_destination_rejects_previously_tracked_content(private_repository):
    destination = private_repository / "data/private"
    destination.mkdir(parents=True)
    (destination / "accidental.json").write_text("{}")
    subprocess.run(["git", "add", "--force", "data/private/accidental.json"], check=True)
    with pytest.raises(DatasetError, match="track"):
        _private_dir(destination)


def test_ignored_private_root_can_link_to_durable_storage(private_repository):
    durable = private_repository.parent / "drive/storage"
    durable.mkdir(parents=True)
    (private_repository / "data").mkdir()
    approved = private_repository / "data/private"
    approved.symlink_to(durable, target_is_directory=True)
    destination = _private_dir(approved / "normalized")
    assert destination == (durable / "normalized").resolve()
    assert destination.is_dir()


def test_nested_symlink_cannot_escape_approved_private_root(private_repository):
    approved = private_repository / "data/private"
    approved.mkdir(parents=True)
    outside = private_repository.parent / "other-storage/private"
    outside.mkdir(parents=True)
    (approved / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(DatasetError, match="private"):
        _private_dir(approved / "escape/normalized")
    assert not (outside / "normalized").exists()


@pytest.fixture
def frozen_numeric_export(tmp_path):
    # These independent numeric fixtures only exercise the empirical-file boundary.
    # Nothing from the real benchmark or a real model appears in these temporary files.
    rows = numeric_rows()
    for row in rows:
        row["data_origin"] = "sleight_bench"
    source = tmp_path / "public_scores.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    labels = {
        row["transcript_id"]: {
            field: row[field]
            for field in ("transcript_id", "scenario_id", "family_id", "split", "label")
        }
        for row in rows
    }
    manifest = {
        "data_origin": "sleight_bench",
        "config": {"bootstrap_samples": 23, "bootstrap_seed": 91},
        "planned_calls": [
            {field: row[field] for field in ("transcript_id", "condition", "repetition")}
            for row in rows
        ],
        "evaluation_labels": list(labels.values()),
    }
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return source, manifest_path


def test_report_defaults_use_frozen_bootstrap_settings(tmp_path, frozen_numeric_export):
    source, manifest = frozen_numeric_export
    metrics = generate_report(source, tmp_path / "report", manifest_path=manifest)
    assert metrics["primary"]["n_bootstrap"] == 23
    assert metrics["primary"]["seed"] == 91


@pytest.mark.parametrize("overrides", [{"n_bootstrap": 24}, {"seed": 92}])
def test_report_rejects_bootstrap_overrides_conflicting_with_manifest(
    tmp_path, frozen_numeric_export, overrides
):
    source, manifest = frozen_numeric_export
    output = tmp_path / "report"
    with pytest.raises(ValueError, match="bootstrap"):
        generate_report(source, output, manifest_path=manifest, **overrides)
    assert not output.exists()


def test_report_accepts_matching_explicit_bootstrap_settings(tmp_path, frozen_numeric_export):
    source, manifest = frozen_numeric_export
    metrics = generate_report(
        source, tmp_path / "report", manifest_path=manifest, n_bootstrap=23, seed=91
    )
    assert metrics["primary"]["n_bootstrap"] == 23
    assert metrics["primary"]["seed"] == 91


def test_reproduction_notebook_uses_manifest_bootstrap_settings(
    tmp_path, frozen_numeric_export, monkeypatch
):
    source, _ = frozen_numeric_export
    monkeypatch.setenv("CONTEXT_AUDIT_SCORES", str(source))
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks/02_reproduce_results.ipynb"
    notebook = json.loads(notebook_path.read_text())
    source_cell = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "BOOTSTRAP_SAMPLES =" in "".join(cell["source"])
    )
    import os

    namespace = {"ROOT": tmp_path, "Path": Path, "os": os}
    exec(compile(source_cell, "offline_reproduction", "exec"), namespace)
    assert namespace["metrics"]["primary"]["n_bootstrap"] == 23
    assert namespace["metrics"]["primary"]["seed"] == 91
