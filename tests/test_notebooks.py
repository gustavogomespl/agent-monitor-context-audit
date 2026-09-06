"""Execute committed notebooks on CPU with real inference explicitly disabled."""

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient


@pytest.mark.parametrize(
    "name,expected",
    [
        ("01_dataset_and_pilot.ipynb", "SYNTHETIC FIXTURE"),
        ("02_reproduce_results.ipynb", "Experiment not executed"),
    ],
)
def test_notebook_default_executes_offline_with_empty_committed_outputs(
    name, expected, monkeypatch
):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("RUN_LIVE", "false")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CONTEXT_AUDIT_SCORES", str(root / "tests" / "absent_numeric_scores.csv"))
    notebook = nbformat.read(root / "notebooks" / name, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.outputs == []
            assert cell.execution_count is None
    executed = NotebookClient(notebook, timeout=120, kernel_name="python3").execute(
        cwd=str(root),
    )
    text = "\n".join(
        output.get("text", "")
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    )
    assert expected in text
