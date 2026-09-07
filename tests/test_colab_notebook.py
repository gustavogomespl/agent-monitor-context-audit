"""The portable Colab notebook must be inert without explicit cell edits."""

import builtins
from pathlib import Path

import nbformat
import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/03_qwen_colab.ipynb"


def _cells():
    assert NOTEBOOK.exists(), "The dedicated Qwen Colab notebook is required"
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.outputs == []
            assert cell.execution_count is None
            yield cell.source


def test_default_colab_notebook_executes_without_side_effects(tmp_path, monkeypatch, capsys):
    import socket
    import subprocess

    def forbidden(*args, **kwargs):
        pytest.fail("Default Run All must not access processes, network, or the filesystem")

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("google.colab", "torch", "vllm", "context_audit")):
            pytest.fail(f"Default notebook imported a live dependency: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUN_SETUP", "true")
    monkeypatch.setenv("RUN_LIVE", "true")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    namespace = {}
    for source in _cells():
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    assert namespace["START_RUN"] is False
    assert namespace["DATA_USE_CONFIRMED"] is False
    assert namespace["RUBRIC_REVIEWED"] is False
    assert namespace["DEVELOPMENT_REVIEWED"] is False
    assert namespace["STAGE"] == "pilot"
    assert namespace["MAX_GPU_HOURS"] == 12.0
    assert namespace["PRIOR_GPU_MINUTES"] == 0
    assert namespace["MAX_COST_USD"] is namespace["GPU_HOURLY_RATE_USD"] is None
    assert list(tmp_path.iterdir()) == []
    assert "Run all" in capsys.readouterr().out


@pytest.mark.parametrize("unsafe_name", ["../escape.txt", "/absolute.txt", ".git/config"])
def test_uploaded_source_zip_cannot_escape_or_overwrite_git(
    tmp_path, monkeypatch, unsafe_name
):
    import zipfile

    monkeypatch.chdir(tmp_path)
    namespace = {}
    for source in _cells():
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("safe.txt", "Independent source fixture")
        target.writestr(unsafe_name, "Independent unsafe-path fixture")
    extraction = tmp_path / "extract"
    extraction.mkdir()
    with pytest.raises(ValueError, match="archive|Archive"):
        namespace["safe_extract"](archive, extraction)
    assert list(extraction.iterdir()) == []
    assert not (tmp_path / "escape.txt").exists()


def test_colab_runtime_pin_is_written_once_and_rejects_dependency_drift(tmp_path, monkeypatch):
    import json

    monkeypatch.chdir(tmp_path)
    namespace = {}
    for source in _cells():
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    assert "verify_runtime_versions" in namespace, "Setup must enforce the installed package pin"
    verify = namespace["verify_runtime_versions"]
    pin = tmp_path / "model-pin.json"
    pin.write_text(json.dumps({"model_revision": "a" * 40}))
    versions = dict.fromkeys(
        ("vllm", "torch", "transformers", "tokenizers", "triton", "safetensors"), "1.0.0"
    )
    result = verify(pin, versions)
    assert result["runtime_versions"] == versions
    saved = pin.read_bytes()
    assert verify(pin, versions) == result
    assert pin.read_bytes() == saved
    with pytest.raises(RuntimeError, match="pinned versions|exploratory"):
        verify(pin, versions | {"transformers": "1.0.1"})
    assert pin.read_bytes() == saved


def test_all_colab_phases_carry_the_saved_runtime_fingerprint(tmp_path, monkeypatch):
    import json

    monkeypatch.chdir(tmp_path)
    namespace = {}
    for source in _cells():
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    configuration = tmp_path / "configuration"
    configuration.mkdir()
    versions = dict.fromkeys(
        ("vllm", "torch", "transformers", "tokenizers", "triton", "safetensors"), "1.0.0"
    )
    pin = {
        "model_id": "Qwen/Qwen3.8-27B",
        "model_revision": "a" * 40,
        "vllm_version": "0.28.0",
        "runtime_versions": versions,
    }
    (configuration / "model-pin.json").write_text(json.dumps(pin))
    namespace.update(
        SETUP_READY=True,
        DRIVE_ROOT=tmp_path,
        DATA_USE_CONFIRMED=True,
        RUBRIC_REVIEWED=True,
    )
    assert namespace["EXPERIMENT_VERSION"] == "summary-v4"
    assert namespace["REPO"] == Path("/content/agent-monitor-context-audit-summary-v4")
    namespace["prepare_version_workspace"]()
    configuration = namespace["source_workspace"]() / "configuration"
    for phase in ("pilot", "development", "test"):
        config = namespace["configured_phase"](phase)
        assert config.qwen.runtime_versions == versions
        assert config.structured_summary_mode == "schema_citations_v1"
        assert config.token_maximum == 2048 and config.summary_max_tokens == 3200
        assert config.token_fraction == 0.25 and config.token_minimum == 128
        assert config.qwen.gpu_budget_hours == 12.0
        assert config.qwen.gpu_hourly_rate_usd is None
        assert config.timeout_seconds == 300
        saved = json.loads((configuration / f"{phase}-summary-v4.json").read_text())
        assert saved["qwen"]["runtime_versions"] == versions
        assert saved["qwen"]["gpu_budget_hours"] == 12.0


def test_readme_and_guide_link_the_hosted_notebook():
    root = NOTEBOOK.parents[1]
    link = (
        "https://colab.research.google.com/github/gustavogomespl/"
        "agent-monitor-context-audit/blob/pilot/notebooks/03_qwen_colab.ipynb"
    )
    for name in ("README.md", "docs/qwen_colab.md"):
        assert link in (root / name).read_text(), name
