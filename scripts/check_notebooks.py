"""Execute empty-output notebooks offline; save executed copies only to private storage."""

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

os.environ["RUN_LIVE"] = "false"
root = Path(__file__).resolve().parents[1]
output = root / "runs/private/notebook_checks"
output.mkdir(parents=True, exist_ok=True)
for path in sorted((root / "notebooks").glob("*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    if any(cell.get("outputs") for cell in notebook.cells):
        raise ValueError(f"{path.name}: committed notebook outputs must be empty")
    NotebookClient(
        notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(root)}}
    ).execute()
    nbformat.write(notebook, output / path.name)
    print(f"{path.name}: valid and executed offline")
