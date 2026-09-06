"""Guided Colab workflows exercise synthetic boundaries; no model or real data."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/colab_bootstrap.py"


def bootstrap(**values):
    ns = runpy.run_path(str(SCRIPT))
    # Functions share the original run_path globals, not its returned copy.
    ns = ns["run_guided"].__globals__
    ns.update(values)
    return ns


@pytest.fixture
def workflow(tmp_path):
    calls = []

    class Allocation:
        def __init__(self, *args):
            calls.append("budget")

        def checkpoint(self):
            return {"remaining_seconds": 40000}

        def close(self):
            calls.append("account")

        def remaining(self):
            return 40000

    ns = bootstrap(
        START_RUN=True, STAGE="pilot", DATA_USE_CONFIRMED=True,
        RUBRIC_REVIEWED=True, DEVELOPMENT_REVIEWED=False,
        PRIOR_GPU_MINUTES=0, DRIVE_ROOT=tmp_path, REPO=tmp_path,
        MAX_GPU_HOURS=12.0, STARTUP_TIMEOUT_SECONDS=1800,
        mount_workspace=lambda: calls.append("mount"),
        check_gpu=lambda: calls.append("hardware"),
        prepare_source=lambda: calls.append("source"),
        install_runtime=lambda: calls.append("install"),
        acquire_data=lambda: calls.append("acquire"),
        NotebookAllocation=Allocation,
        configured_phase=lambda phase: SimpleNamespace(run_dir=f"runs/private/qwen-{phase}"),
        phase_complete=lambda config: False,
        execute_phase=lambda config, seconds: (
            calls.append(config.run_dir) or {"status": "executed"}
        ),
        export_phase=lambda phase, seconds: calls.append("report-" + phase) or {},
        freeze_reviewed=lambda: calls.append("freeze"),
        release_gpu=lambda: calls.append("disconnect"),
        record_status=lambda *args, **kwargs: calls.append("status"),
    )
    return ns, calls


def test_one_start_runs_pilot_through_report_and_disconnect(workflow):
    ns, calls = workflow
    result = ns["run_guided"]()
    assert result["status"] == "executed"
    assert calls == [
        "hardware", "mount", "budget", "source", "install", "acquire",
        "runs/private/qwen-pilot", "report-pilot", "status", "account", "disconnect",
    ]


def test_development_runs_pilot_first_and_never_test(workflow):
    ns, calls = workflow
    ns["STAGE"] = "development"
    ns["run_guided"]()
    assert calls.index("report-pilot") < calls.index("runs/private/qwen-development")
    assert "freeze" not in calls and "runs/private/qwen-test" not in calls


def test_test_requires_review_before_mounting_or_allocating(workflow):
    ns, calls = workflow
    ns["STAGE"] = "test"
    with pytest.raises(ValueError, match="review"):
        ns["run_guided"]()
    assert calls == []


def test_reviewed_test_freezes_before_launch(workflow):
    ns, calls = workflow
    ns.update(STAGE="test", DEVELOPMENT_REVIEWED=True)
    ns["run_guided"]()
    assert calls.index("freeze") < calls.index("runs/private/qwen-test")


@pytest.mark.parametrize("consent", ["DATA_USE_CONFIRMED", "RUBRIC_REVIEWED"])
def test_missing_consent_has_no_setup_side_effects(workflow, consent):
    ns, calls = workflow
    ns[consent] = False
    with pytest.raises(ValueError, match="confirm"):
        ns["run_guided"]()
    assert calls == []


def test_completed_phase_is_exported_without_loading_model(workflow):
    ns, calls = workflow
    ns["phase_complete"] = lambda config: True
    ns["run_guided"]()
    assert "runs/private/qwen-pilot" not in calls and "report-pilot" in calls


def test_failed_pilot_does_not_proceed_to_development(workflow):
    ns, calls = workflow
    ns["STAGE"] = "development"
    ns["execute_phase"] = lambda *args: {"status": "partial_or_failed"}
    result = ns["run_guided"]()
    assert result["status"] == "partial_or_failed"
    assert "report-pilot" in calls and "runs/private/qwen-development" not in calls
    assert calls[-2:] == ["account", "disconnect"]


def test_token_only_context_failure_restarts_once_before_reporting(workflow):
    ns, calls = workflow
    selection_calls = iter([False, True])  # Fresh workspace, then completed context preflight.
    ns["prepare_context_window"] = lambda: next(selection_calls)
    runs = []

    def execute(config, seconds):
        runs.append((config, seconds))
        return {"status": "partial_or_failed"}

    ns["execute_phase"] = execute
    result = ns["run_guided"]()
    assert result["status"] == "partial_or_failed"
    assert len(runs) == 2  # No retry loop after the one context adjustment.
    assert all(seconds == 39850 for _, seconds in runs)
    assert calls.count("report-pilot") == 1
    assert calls[-2:] == ["account", "disconnect"]


def test_context_restart_rechecks_remaining_allocation(workflow):
    ns, calls = workflow
    selections = iter([False, True])
    ns["prepare_context_window"] = lambda: next(selections)

    def execute(config, seconds):
        calls.append("first-run")
        ns["NotebookAllocation"].remaining = lambda self: 100
        return {"status": "partial_or_failed"}

    ns["execute_phase"] = execute
    with pytest.raises(RuntimeError, match="insufficient GPU time"):
        ns["run_guided"]()
    assert calls.count("first-run") == 1
    assert calls[-2:] == ["account", "disconnect"]


@pytest.mark.parametrize("boundary", ["check_gpu", "prepare_source", "install_runtime",
                                     "acquire_data", "execute_phase", "export_phase"])
def test_every_failure_releases_gpu_and_preserves_failure_status(workflow, boundary):
    ns, calls = workflow

    def fail(*args):
        raise RuntimeError("Independent synthetic failure")

    ns[boundary] = fail
    with pytest.raises(RuntimeError, match="private|saved"):
        ns["run_guided"]()
    if boundary == "check_gpu":
        # A CPU runtime fails before Drive is mounted or any budget receipt exists.
        assert calls == ["status", "disconnect"]
    else:
        assert calls[-2:] == ["account", "disconnect"]
        assert "status" in calls


def test_existing_initial_debit_is_restored_without_editing_form(tmp_path, monkeypatch):
    import threading

    from context_audit.colab import initialize_gpu_budget

    monkeypatch.setattr(threading, "Timer", lambda *args: SimpleNamespace(
        start=lambda: None, cancel=lambda: None,
    ))
    run = tmp_path / "runs-private/qwen-pilot"
    original = initialize_gpu_budget(run, 12, previously_used_seconds=1800)
    ns = bootstrap(MAX_GPU_HOURS=12.0, PRIOR_GPU_MINUTES=0)
    allocation = ns["NotebookAllocation"](tmp_path, 0, lambda: None)
    receipt = allocation.checkpoint()
    assert receipt["committed_seconds"] >= original["committed_seconds"]
    assert receipt["remaining_seconds"] <= 41400
    allocation.close()
    pin = json.loads((run.parent / "gpu_budget/manifests/limit.json").read_text())
    assert pin == {"limit_seconds": 43200.0, "previously_used_seconds": 1800}


def test_overhead_excludes_time_already_charged_by_supervisor(tmp_path, monkeypatch):
    import threading
    import time

    from context_audit.colab import initialize_gpu_budget, record_external_gpu_time

    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(threading, "Timer", lambda *args: SimpleNamespace(
        start=lambda: None, cancel=lambda: None,
    ))
    run = tmp_path / "runs-private/qwen-pilot"
    initialize_gpu_budget(run, 12, previously_used_seconds=1800)
    ns = bootstrap()
    allocation = ns["NotebookAllocation"](tmp_path, clock.now, lambda: None)
    clock.now += 60
    assert allocation.checkpoint()["committed_seconds"] == 1860
    # Independent simulated supervisor receipt: 120 seconds within a 150-second interval.
    record_external_gpu_time(run, 12, usage_id="synthetic-managed", elapsed_seconds=120,
                             confirmed=True)
    clock.now += 150
    assert allocation.checkpoint()["committed_seconds"] == 2010
    allocation.close()
    assert ns["read_budget"](tmp_path)["committed_seconds"] == 2010


def test_unknown_notebook_end_time_blocks_automatic_resumption(tmp_path):
    folder = tmp_path / "runs-private/notebook-sessions"
    folder.mkdir(parents=True)
    (folder / "synthetic.json").write_text(json.dumps({"status": "running"}))
    ns = bootstrap()
    with pytest.raises(ValueError, match="end time is unknown"):
        ns["NotebookAllocation"](tmp_path, 0, lambda: None)


@pytest.mark.parametrize("previously_settled", [False, True])
def test_known_setup_failure_overhead_is_recovered_once(
    tmp_path, monkeypatch, previously_settled
):
    import threading
    import time

    from context_audit.colab import initialize_gpu_budget, record_external_gpu_time

    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(threading, "Timer", lambda *args: SimpleNamespace(
        start=lambda: None, cancel=lambda: None,
    ))
    run = tmp_path / "runs-private/qwen-pilot"
    initialize_gpu_budget(run, 12, previously_used_seconds=1800)
    history = tmp_path / "runs-private/notebook-sessions"
    history.mkdir()
    (history / "synthetic.json").write_text(json.dumps({
        "status": "stopped_unaccounted", "unaccounted_seconds": 60,
    }))
    if previously_settled:
        record_external_gpu_time(run, 12, usage_id="notebook-synthetic-recovery",
                                 elapsed_seconds=60, confirmed=True)
    allocation = bootstrap()["NotebookAllocation"](tmp_path, 100, lambda: None)
    assert allocation.checkpoint()["committed_seconds"] == 1860
    allocation.close()
    assert initialize_gpu_budget(run, 12, previously_used_seconds=1800)["committed_seconds"] == 1860


def test_completed_reuse_checks_signature_and_rejects_modified_scores(tmp_path, monkeypatch):
    import hashlib

    import context_audit.runner as runner
    from context_audit.storage import digest

    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "data/private/manifest.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_text(json.dumps({"schema_version": 1, "private_fixture_field": "excluded"}))
    config = SimpleNamespace(run_dir="runs/private/qwen-pilot", dataset_dir="data/private",
                             model_dump=lambda: {"independent": "fixture"})

    def signature(config, manifest):
        assert manifest == {"schema_version": 1}
        return {"code_hash": "a" * 64, "prompts": {}, "dataset_manifest_hash": digest(manifest)}

    monkeypatch.setattr(runner, "protocol_signature", signature)
    directory = tmp_path / config.run_dir
    (directory / "manifests").mkdir(parents=True)
    scores = directory / "scores.csv"
    scores.write_text("Independent synthetic checksum fixture\n")
    (directory / "manifests/run.json").write_text(json.dumps({
        "code_hash": "a" * 64, "prompt_hashes": {}, "config": config.model_dump(),
        "dataset_manifest_hash": digest({"schema_version": 1}),
        "run_id": "synthetic", "planned_calls": [{}],
    }))
    (directory / "manifests/completion.json").write_text(json.dumps({
        "status": "executed", "run_id": "synthetic", "rows": 1,
        "successful_rows": 1, "expected_rows": 1,
        "scores_sha256": hashlib.sha256(scores.read_bytes()).hexdigest(),
    }))
    ns = bootstrap()
    assert ns["phase_complete"](config) is True
    scores.write_text("Modified synthetic scores\n")
    with pytest.raises(ValueError, match="integrity review"):
        ns["phase_complete"](config)


def test_not_started_output_lists_unchecked_form_fields(workflow, capsys):
    ns, calls = workflow
    ns.update(START_RUN=False, DATA_USE_CONFIRMED=False)
    assert ns["run_guided"]() == {"status": "not_started"}
    out = capsys.readouterr().out
    assert "[ ] DATA_USE_CONFIRMED" in out
    assert "[x] RUBRIC_REVIEWED" in out
    assert "[ ] START_RUN" in out
    assert "DEVELOPMENT_REVIEWED" not in out
    assert "Run all" in out
    assert calls == []


def test_not_started_output_includes_development_review_only_for_test(workflow, capsys):
    ns, calls = workflow
    ns.update(START_RUN=False, STAGE="test")
    ns["run_guided"]()
    assert "[ ] DEVELOPMENT_REVIEWED" in capsys.readouterr().out


def test_start_announces_stage_model_and_budget_before_side_effects(workflow, capsys):
    ns, calls = workflow

    def fail():
        raise RuntimeError("Independent synthetic failure")

    ns["check_gpu"] = fail
    with pytest.raises(RuntimeError):
        ns["run_guided"]()
    out = capsys.readouterr().out
    assert "Stage: pilot" in out
    assert "Qwen/Qwen3.8-27B" in out
    assert "12 GPU hours" in out
    assert out.index("Stage: pilot") < out.index("Independent synthetic failure")


def test_failure_output_names_the_error_class_and_message(workflow, capsys):
    ns, calls = workflow

    def fail():
        raise RuntimeError("Independent synthetic failure")

    ns["prepare_source"] = fail
    with pytest.raises(RuntimeError, match="Independent synthetic failure"):
        ns["run_guided"]()
    assert "RuntimeError: Independent synthetic failure" in capsys.readouterr().out


def test_validation_error_details_stay_out_of_the_notebook_output(workflow, capsys):
    from context_audit.runtime_models import MonitorDecision

    ns, calls = workflow
    with pytest.raises(ValueError) as caught:
        MonitorDecision.model_validate({"suspicion_score": "Independent synthetic private text"})
    error = caught.value
    assert "synthetic private text" in str(error)

    def fail():
        raise error

    ns["acquire_data"] = fail
    with pytest.raises(RuntimeError) as raised:
        ns["run_guided"]()
    out = capsys.readouterr().out
    assert "ValidationError" in out
    assert "synthetic private text" not in out
    assert "synthetic private text" not in str(raised.value)


def test_partial_run_prints_progress_counts_and_exit_code(workflow, capsys, tmp_path):
    ns, calls = workflow
    run_dir = tmp_path / "runs/private/qwen-pilot"
    (run_dir / "manifests").mkdir(parents=True)
    (run_dir / "scores.csv").write_text(
        "transcript_id,status\nt_a,ok\nt_b,api_error\nt_c,api_error\n"
    )
    (run_dir / "manifests/completion.json").write_text(json.dumps({
        "status": "partial_or_failed", "rows": 3, "successful_rows": 1, "expected_rows": 24,
    }))
    ns["execute_phase"] = lambda config, seconds: {"status": "partial_or_failed", "exit_code": 1}
    assert ns["run_guided"]()["status"] == "partial_or_failed"
    out = capsys.readouterr().out
    assert "exit code 1" in out
    assert "1/24" in out
    assert "api_error: 2" in out
    assert "ok: 1" in out


def test_partial_run_surfaces_the_runner_error_line(workflow, capsys, tmp_path):
    ns, calls = workflow
    logs = tmp_path / "runs-private/qwen-pilot/gpu_sessions/runner_logs"
    logs.mkdir(parents=True)
    (logs / "synthetic.json").write_text(
        "progress line\ncontext-audit: 3 transcripts exceed context; revise scope before scoring\n"
    )
    ns["execute_phase"] = lambda config, seconds: {"status": "partial_or_failed", "exit_code": 2}
    ns["run_guided"]()
    assert "3 transcripts exceed context" in capsys.readouterr().out


@pytest.mark.parametrize("details,expected", [
    ("RuntimeError: Torch compiled with different CUDA versions than TorchAudio",
     "Torch and TorchAudio"),
    ("torch.OutOfMemoryError: CUDA out of memory", "GPU memory"),
    ("topk_topp_sampler.py\nRuntimeError: FlashInfer requires GPUs with sm75 or higher",
     "FlashInfer sampler"),
    ("ValueError: Previous notebook end time is unknown; reconcile", "uncertain"),
    ("progress\ncontext-audit: 3 transcripts exceed context; revise scope before scoring\n",
     "3 transcripts exceed context"),
])
def test_failure_hints_explain_known_diagnostics(details, expected):
    hints = bootstrap()["failure_hints"](details)
    assert any(expected in hint for hint in hints), hints


def test_failure_hints_never_repeat_validation_error_details():
    details = (
        "context-audit: 1 validation error for TranscriptInput "
        "input_value='Independent synthetic private text'\n"
    )
    hints = bootstrap()["failure_hints"](details)
    assert hints
    assert all("synthetic private text" not in hint for hint in hints)
    assert any("validation error" in hint for hint in hints)


def test_missing_gpu_names_the_runtime_type_fix(monkeypatch):
    import subprocess

    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="Change runtime type"):
        bootstrap()["check_gpu"]()


def test_small_gpu_report_names_the_detected_card(monkeypatch):
    import subprocess

    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="Tesla T4, 15360\n")
    )
    with pytest.raises(RuntimeError, match="Tesla T4"):
        bootstrap()["check_gpu"]()


def test_install_commands_are_quiet_and_pin_the_saved_runtime():
    commands = bootstrap()["install_commands"]({
        "vllm_version": "0.28.0", "runtime_versions": {"torch": "2.13.0", "vllm": "0.28.0"},
    })
    assert len(commands) == 2
    for command in commands:
        assert command[:4] == [command[0], "-m", "pip", "install"]
        assert "-q" in command
    assert {"vllm==0.28.0", "torch==2.13.0"} <= set(commands[0])
    assert "transformers>=5.8.0,<6" in commands[0]
    assert commands[1][-2:] == ["-e", ".[data]"]
