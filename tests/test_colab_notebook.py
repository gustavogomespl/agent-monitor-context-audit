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
    assert namespace["RUN_SETUP"] is namespace["RUN_LIVE"] is False
    assert namespace["RUN_ACQUIRE"] is namespace["RUN_FREEZE"] is False
    assert namespace["RUN_TEST"] is namespace["RUN_ANALYSIS"] is False
    assert namespace["RUN_RECONCILE"] is False
    assert namespace["MAX_GPU_HOURS"] == 12.0
    assert namespace["GPU_ALREADY_USED_HOURS"] == 0.0
    assert namespace["GPU_ADDITIONAL_USED_HOURS"] == 0.0
    assert namespace["GPU_ADDITIONAL_USAGE_ID"] == ""
    assert namespace["MAX_COST_USD"] is namespace["GPU_HOURLY_RATE_USD"] is None
    assert namespace["DISCONNECT_AFTER_RUN"] is True
    assert list(tmp_path.iterdir()) == []
    assert "No setup or generation" in capsys.readouterr().out


def test_colab_live_entry_requires_completed_explicit_setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = {}
    cells = list(_cells())
    for source in cells:
        if "# LIVE_EXECUTION" in source:
            namespace["RUN_LIVE"] = True
            with pytest.raises(RuntimeError, match="RUN_SETUP"):
                exec(compile(source, str(NOTEBOOK), "exec"), namespace)
            break
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    else:
        pytest.fail("Notebook lacks its managed live execution phase")


def test_colab_test_phase_requires_separate_opt_in(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for source in _cells():
        if "# LIVE_EXECUTION" in source:
            namespace.update(RUN_LIVE=True, PHASE="test", SETUP_READY=True)
            with pytest.raises(RuntimeError, match="RUN_TEST"):
                exec(compile(source, str(NOTEBOOK), "exec"), namespace)
            break
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)


def test_colab_freeze_cannot_share_a_generation_phase(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for source in _cells():
        if "# LIVE_EXECUTION" in source:
            namespace.update(RUN_LIVE=True, RUN_FREEZE=True, SETUP_READY=True)
            with pytest.raises(RuntimeError, match="separate phases"):
                exec(compile(source, str(NOTEBOOK), "exec"), namespace)
            break
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)


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


def test_colab_session_reconciliation_requires_separate_confirmation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for source in _cells():
        if "# SESSION_RECONCILIATION" in source:
            namespace.update(RUN_RECONCILE=True, SETUP_READY=True)
            with pytest.raises(ValueError, match="CONFIRM_RECONCILIATION"):
                exec(compile(source, str(NOTEBOOK), "exec"), namespace)
            break
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    else:
        pytest.fail("The notebook must expose opt-in interrupted-session reconciliation")


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
    for phase in ("pilot", "development", "test"):
        config = namespace["configured_phase"](phase)
        assert config.qwen.runtime_versions == versions
        assert config.qwen.gpu_budget_hours == 12.0
        assert config.qwen.gpu_hourly_rate_usd is None
        assert config.timeout_seconds == 300
        saved = json.loads((configuration / f"{phase}.json").read_text())
        assert saved["qwen"]["runtime_versions"] == versions
        assert saved["qwen"]["gpu_budget_hours"] == 12.0


@pytest.mark.parametrize("additional_hours", [0.0, 0.25])
def test_colab_live_debits_prior_gpu_allocation_before_launch_without_usd(
    tmp_path, monkeypatch, additional_hours
):
    from types import SimpleNamespace

    import context_audit.colab as runtime

    monkeypatch.chdir(tmp_path)
    namespace = {}
    sources = list(_cells())
    for source in sources:
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    config = SimpleNamespace(run_dir="runs/private/qwen-pilot")
    calls = []

    def initialize(run_dir, max_gpu_hours, previously_used_seconds):
        calls.append(("initialize", run_dir, max_gpu_hours, previously_used_seconds))
        return {"remaining_seconds": 41400}

    def run(actual_config, **kwargs):
        assert actual_config is config
        expected = [("initialize", Path(config.run_dir), 12.0, 1800.0)]
        if additional_hours:
            expected.append(("external", Path(config.run_dir), {
                "max_gpu_hours": 12.0, "usage_id": "observed-runtime-two-setup",
                "elapsed_seconds": 900.0, "confirmed": True,
            }))
        assert calls == expected
        calls.append(("run", kwargs))
        return {"status": "independent_synthetic_fixture"}

    monkeypatch.setattr(runtime, "initialize_gpu_budget", initialize, raising=False)
    monkeypatch.setattr(
        runtime, "record_external_gpu_time",
        lambda run_dir, **kwargs: calls.append(("external", run_dir, kwargs)), raising=False,
    )
    monkeypatch.setattr(runtime, "run_colab_experiment", run)
    namespace.update(
        RUN_LIVE=True,
        SETUP_READY=True,
        DISCONNECT_AFTER_RUN=False,
        REPO=tmp_path,
        GPU_ALREADY_USED_HOURS=0.5,
        GPU_ADDITIONAL_USED_HOURS=additional_hours,
        GPU_ADDITIONAL_USAGE_ID="observed-runtime-two-setup",
        configured_phase=lambda phase: config,
    )
    live_source = next(source for source in sources if "# LIVE_EXECUTION" in source)
    exec(compile(live_source, str(NOTEBOOK), "exec"), namespace)
    assert calls[-1] == (
        "run",
        {"max_cost_usd": None, "session_max_seconds": 3600, "startup_timeout_seconds": 900},
    )


def test_colab_disconnects_without_gpu_postprocessing_after_interruption(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    import context_audit.cli as cli
    import context_audit.colab as runtime
    import context_audit.reporting as reporting

    namespace = {}
    sources = list(_cells())
    for source in sources:
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("Live teardown must release the GPU before CPU exports and reports")

    def interrupted(*args, **kwargs):
        raise RuntimeError("Synthetic managed interruption")

    monkeypatch.setattr(runtime, "initialize_gpu_budget", lambda *args, **kwargs: {})
    monkeypatch.setattr(runtime, "run_colab_experiment", interrupted)
    monkeypatch.setattr(cli, "export_results", forbidden)
    monkeypatch.setattr(reporting, "generate_report", forbidden)
    colab = ModuleType("google.colab")
    colab.runtime = SimpleNamespace(unassign=lambda: calls.append("disconnect"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    namespace.update(
        RUN_LIVE=True, SETUP_READY=True, REPO=tmp_path,
        configured_phase=lambda phase: SimpleNamespace(run_dir="runs/private/qwen-pilot"),
    )
    live = next(source for source in sources if "# LIVE_EXECUTION" in source)
    with pytest.raises(RuntimeError, match="Synthetic managed interruption"):
        exec(compile(live, str(NOTEBOOK), "exec"), namespace)
    assert calls == ["disconnect"]


def test_colab_reconciliation_accepts_verified_time_without_usd_cap(tmp_path, monkeypatch):
    import context_audit.colab as runtime

    monkeypatch.chdir(tmp_path)
    namespace = {}
    sources = list(_cells())
    for source in sources:
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    calls = []
    monkeypatch.setattr(
        runtime, "reconcile_gpu_session", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    namespace.update(
        RUN_RECONCILE=True,
        CONFIRM_RECONCILIATION=True,
        SETUP_READY=True,
        REPO=tmp_path,
        RECONCILE_SESSION_ID="independent-synthetic-session",
        RECONCILE_ELAPSED_SECONDS=1200,
    )
    source = next(source for source in sources if "# SESSION_RECONCILIATION" in source)
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    assert calls == [
        (
            (Path("runs/private/qwen-pilot"), "independent-synthetic-session"),
            {"elapsed_seconds": 1200, "max_cost_usd": None, "confirmed": True},
        )
    ]
