"""Probe notebook opt-in routing with a synthetic, non-generating CLI boundary."""

import json
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("kernel_directory", [".", "notebooks"])
@pytest.mark.parametrize("key_source", ["environment", "dotenv"])
def test_live_notebook_uses_repository_paths_and_local_key(
    tmp_path, monkeypatch, kernel_directory, key_source
):
    import context_audit.cli as cli

    notebook = json.loads(
        (Path(__file__).resolve().parents[1] / "notebooks/01_dataset_and_pilot.ipynb").read_text()
    )
    live_cell = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "RUN_LIVE =" in "".join(cell["source"])
    )
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "data/private").mkdir(parents=True)
    monkeypatch.setenv("RUN_LIVE", "true")
    monkeypatch.setenv("MAX_COST_USD", "0.25")
    synthetic_key = "independent-synthetic-entry-key"
    if key_source == "dotenv":
        # Register cleanup even when dotenv later adds a previously absent key.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        (tmp_path / ".env").write_text(f"ANTHROPIC_API_KEY={synthetic_key}\n")
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", synthetic_key)
        (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=do-not-override-explicit-key\n")
    start_directory = tmp_path / kernel_directory
    monkeypatch.chdir(start_directory)
    invoked = []

    def non_generating_cli_boundary(arguments):
        # Normal CLI paths are repository-relative. Do not touch real data or API.
        assert Path.cwd() == tmp_path
        assert os.environ["ANTHROPIC_API_KEY"] == synthetic_key
        assert "--live" in arguments
        assert arguments[arguments.index("--max-cost-usd") + 1] == "0.25"
        assert Path(arguments[arguments.index("--config") + 1]) == tmp_path / "configs/pilot.yaml"
        invoked.append(True)
        return 0

    monkeypatch.setattr(cli, "main", non_generating_cli_boundary)
    config = dict(
        dataset_dir="data/private",
        monitor_model="independent-fixture-monitor",
        summarizer_model="independent-fixture-summary",
        data_use_confirmed=True,
        rubric_reviewed=True,
    )
    exec(
        compile(live_cell, "notebook_live_cell", "exec"),
        dict(os=os, ROOT=tmp_path, config=config),
    )
    assert invoked == [True]
    assert Path.cwd() == start_directory
