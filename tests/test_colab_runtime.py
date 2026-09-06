"""GPU lifecycle contracts exercised without GPU allocation or model generation."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest

from context_audit.provider import BudgetLedger
from context_audit.runtime_models import AuditConfig, QwenConfig
from context_audit.storage import PrivateStore


def qwen_config(**kwargs):
    return AuditConfig(
        provider="qwen_local",
        monitor_model="Qwen/Qwen3.8-27B",
        summarizer_model="Qwen/Qwen3.8-27B",
        qwen=QwenConfig(model_revision="a" * 40, gpu_hourly_rate_usd=2),
        **kwargs,
    )


def test_managed_server_has_no_tools_and_pins_engine_settings():
    from context_audit.colab import vllm_command

    command = vllm_command(qwen_config())
    assert command[command.index("--revision") + 1] == "a" * 40
    assert command[command.index("--tokenizer-revision") + 1] == "a" * 40
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert "--no-enable-prefix-caching" in command
    assert "--no-enable-log-requests" in command
    assert "--default-chat-template-kwargs" in command
    assert not any("tool" in arg for arg in command)


def test_unapproved_run_never_launches_gpu(monkeypatch):
    import context_audit.colab as colab

    monkeypatch.setattr(colab.subprocess, "Popen", lambda *a, **k: pytest.fail("No launch"))
    with pytest.raises(ValueError, match="data|review"):
        colab.run_colab_experiment(qwen_config(), max_cost_usd=1)


def test_cost_reconciliation_preserves_request_costs_and_updates_checksum(tmp_path):
    from test_metrics import numeric_rows

    from context_audit.colab import finalize_gpu_costs
    from context_audit.provider import BudgetLedger
    from context_audit.storage import PrivateStore

    rows = numeric_rows()
    pd.DataFrame(rows).to_csv(tmp_path / "scores.csv", index=False)
    store = PrivateStore(tmp_path)
    store.put("manifests", "completion", {"status": "executed", "rows": len(rows)})
    ledger = BudgetLedger(PrivateStore(tmp_path / "gpu_sessions"), 5)
    ledger.reserve("session1", 4)
    ledger.settle("session1", 2)
    result = finalize_gpu_costs(tmp_path, 5)
    frame = pd.read_csv(tmp_path / "scores.csv")
    assert frame.cost_usd.sum() == pytest.approx(2)
    assert frame.summary_cost_usd.sum() == pytest.approx(len(rows) * 0.01)
    assert frame.monitor_cost_usd.sum() == pytest.approx(len(rows) * 0.02)
    assert frame.infrastructure_cost_usd.nunique() == 1
    assert result["managed_session_cost_usd"] == 2
    first = frame.copy()
    finalize_gpu_costs(tmp_path, 5)
    assert pd.read_csv(tmp_path / "scores.csv").equals(first)
    completion = json.loads((tmp_path / "manifests/completion.json").read_text())
    assert len(completion["scores_sha256"]) == 64


def test_uncertain_session_requires_explicit_reconciliation(tmp_path):
    from context_audit.colab import reconcile_gpu_session
    from context_audit.provider import BudgetLedger
    from context_audit.storage import PrivateStore

    store = PrivateStore(tmp_path / "gpu_sessions")
    ledger = BudgetLedger(store, 2)
    ledger.reserve("session1", 2)
    store.put("sessions", "session1", {"hourly_rate_usd": 2})
    with pytest.raises(ValueError, match="confirm"):
        reconcile_gpu_session(tmp_path, "session1", elapsed_seconds=900, max_cost_usd=2)
    reconcile_gpu_session(tmp_path, "session1", elapsed_seconds=900, max_cost_usd=2, confirmed=True)
    assert BudgetLedger(store, 2).committed == pytest.approx(0.5)


def test_direct_qwen_run_requires_active_managed_session(tmp_path, monkeypatch):
    from context_audit.colab import require_managed_session

    monkeypatch.delenv("CONTEXT_AUDIT_GPU_SESSION", raising=False)
    with pytest.raises(ValueError, match="supervisor|managed"):
        require_managed_session(qwen_config(run_dir=str(tmp_path)), 1)


def test_teardown_kills_group_even_if_leader_already_exited(monkeypatch):
    import signal
    from types import SimpleNamespace

    import context_audit.colab as colab

    signals = []
    leader = SimpleNamespace(pid=12345, poll=lambda: 0, wait=lambda **kw: 0)
    monkeypatch.setattr(colab.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(colab.time, "sleep", lambda seconds: None)
    colab._stop(leader)
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]


@pytest.fixture
def simulated_lifecycle(tmp_path, monkeypatch):
    """Mock only GPU/process/HTTP/time boundaries; use real session journals."""
    import context_audit.colab as colab

    state = SimpleNamespace(
        clock=100.0, startup_failure=False, worker_exit=0, cleanup_failure=False,
        launched=[], stopped=[], health_requests=[], barrier_read_fds=[],
    )
    config = qwen_config(
        run_dir=str(tmp_path / "qwen-pilot"), data_use_confirmed=True, rubric_reviewed=True,
    )
    config.qwen.gpu_hourly_rate_usd = 3.6
    store = PrivateStore(Path(config.run_dir) / "gpu_sessions")

    def current_ledger():
        if config.qwen.gpu_budget_hours is not None:
            return BudgetLedger(
                PrivateStore(Path(config.run_dir).parent / "gpu_budget"),
                config.qwen.gpu_budget_hours * 3600, unit="seconds",
            )
        return BudgetLedger(store, 1)

    def launch(command, **kwargs):
        ledger = current_ledger()
        if config.qwen.gpu_budget_hours is None:
            assert ledger.committed == pytest.approx(0.06) and not ledger.settled
        else:
            assert "synthetic_session" in ledger.amounts
            assert "synthetic_session" not in ledger.settled
        assert kwargs["start_new_session"] is True
        if colab._LAUNCH_BARRIER in command:
            state.barrier_read_fds.append(os.dup(kwargs["pass_fds"][0]))
            command = command[4:]
        if colab._WATCHDOG in command:
            name, pid = "watchdog", 1001
        elif "vllm.entrypoints.openai.api_server" in command:
            name, pid = "server", 1002
        elif "context_audit.cli" in command:
            name, pid = "worker", 1003
            assert kwargs["env"]["CONTEXT_AUDIT_GPU_SESSION"] == "synthetic_session"
        else:
            pytest.fail("An unexpected child command reached the process boundary")
        state.launched.append(name)
        state.clock += 1
        status = state.worker_exit if name == "worker" else None
        if name == "server" and state.startup_failure:
            status = 1
        return SimpleNamespace(name=name, pid=pid, poll=lambda: status)

    def stop(process):
        if process is None:
            return
        assert "synthetic_session" not in current_ledger().settled
        state.stopped.append(process.name)
        if process.name == "server" and state.cleanup_failure:
            raise OSError("Independent synthetic shutdown uncertainty")
        state.clock += 2

    def health(request):
        assert request.method == "GET"
        assert str(request.url) == "http://127.0.0.1:8000/health"
        state.health_requests.append(request.url.path)
        return httpx.Response(200)

    client_type = httpx.Client

    def client(**kwargs):
        assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
        return client_type(**kwargs, transport=httpx.MockTransport(health))

    monkeypatch.setattr(colab.subprocess, "Popen", launch)
    monkeypatch.setattr(colab, "_stop", stop)
    monkeypatch.setattr(colab.httpx, "Client", client)
    monkeypatch.setattr(colab, "time", SimpleNamespace(
        monotonic=lambda: state.clock, time=lambda: 1000 + state.clock,
        sleep=lambda duration: setattr(state, "clock", state.clock + duration),
    ))
    monkeypatch.setattr(colab.uuid, "uuid4", lambda: SimpleNamespace(hex="synthetic_session"))
    yield colab, config, store, state
    for fd in state.barrier_read_fds:
        os.close(fd)


def test_startup_failure_settles_only_after_confirmed_shutdown(simulated_lifecycle):
    colab, config, store, state = simulated_lifecycle
    state.startup_failure = True
    with pytest.raises(ValueError, match="startup failed"):
        colab._run_managed(config, 1, 60, 30, {"name": "synthetic H100"})
    ledger = BudgetLedger(store, 1)
    assert ledger.settled == {"synthetic_session"}
    assert ledger.committed == pytest.approx(0.006)
    assert state.launched == ["watchdog", "server"] and not state.health_requests
    assert state.stopped == ["server", "watchdog"]
    assert store.get("watchdogs", "synthetic_session")["active"] is False
    assert store.get("sessions", "synthetic_session")["elapsed_seconds"] == 6


@pytest.mark.parametrize("exit_code,status", [(0, "executed"), (1, "partial_or_failed")])
def test_worker_result_preserves_status_and_settles_session(simulated_lifecycle, exit_code, status):
    colab, config, store, state = simulated_lifecycle
    state.worker_exit = exit_code
    result = colab._run_managed(config, 1, 60, 30, {"name": "synthetic H100"})
    assert result["status"] == status and result["exit_code"] == exit_code
    assert result["gpu_costs"]["managed_session_cost_usd"] == pytest.approx(0.009)
    ledger = BudgetLedger(store, 1)
    assert ledger.settled == {"synthetic_session"} and ledger.committed == pytest.approx(0.009)
    assert state.launched == ["watchdog", "server", "worker"]
    assert state.stopped == ["worker", "server", "watchdog"]
    assert state.health_requests == ["/health"]
    session = store.get("sessions", "synthetic_session")
    assert session["elapsed_seconds"] == 9 and session["exit_code"] == exit_code


def test_uncertain_cleanup_keeps_reservation_and_watchdog_active(simulated_lifecycle):
    colab, config, store, state = simulated_lifecycle
    state.cleanup_failure = True
    with pytest.raises(OSError, match="shutdown uncertainty"):
        colab._run_managed(config, 1, 60, 30, {"name": "synthetic H100"})
    ledger = BudgetLedger(store, 1)
    assert not ledger.settled and ledger.committed == pytest.approx(0.06)
    assert state.stopped == ["worker", "server"]
    watchdog = store.get("watchdogs", "synthetic_session")
    assert watchdog == {"active": True, "children": [1002, 1003]}
    assert "elapsed_seconds" not in store.get("sessions", "synthetic_session")
    assert [entry["action"] for entry in store.read_journal("budget")] == ["reserve"]


@pytest.mark.parametrize("release", [b"", b"x"])
def test_child_barrier_exits_without_executing_on_eof_or_invalid_release(tmp_path, release):
    import context_audit.colab as colab

    assert hasattr(colab, "_LAUNCH_BARRIER"), "Children need a release barrier before execution"
    marker = tmp_path / "must_not_exist"
    read_fd, write_fd = os.pipe()
    process = None
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", colab._LAUNCH_BARRIER, str(read_fd), sys.executable,
             "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).touch()", str(marker)],
            pass_fds=(read_fd,), start_new_session=True,
        )
        if release:
            os.write(write_fd, release)
        os.close(write_fd)
        write_fd = None
        assert process.wait(timeout=5) == 0
        assert not marker.exists()
    finally:
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_registered_child_preserves_pid_group_environment_and_output(tmp_path):
    import context_audit.colab as colab

    assert hasattr(colab, "_launch_registered"), "Register children before releasing execution"
    store = PrivateStore(tmp_path)
    control = {"active": True, "children": []}
    store.put("watchdogs", "fixture", control)
    script = (
        "import json, os, sys; "
        "state = json.load(open(sys.argv[1])); "
        "assert os.getpid() in state['children']; "
        "assert os.getpgrp() == os.getpid(); "
        "print(os.environ['INDEPENDENT_BARRIER_FIXTURE'])"
    )
    process = colab._launch_registered(
        [sys.executable, "-c", script, str(store.path("watchdogs", "fixture"))],
        store=store, session_id="fixture", control=control,
        env={**os.environ, "INDEPENDENT_BARRIER_FIXTURE": "registered before execution"},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0 and stderr == b""
        assert stdout == b"registered before execution\n"
        assert store.get("watchdogs", "fixture")["children"] == [process.pid]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_failed_child_registration_closes_release_pipe_without_executing(tmp_path, monkeypatch):
    import context_audit.colab as colab

    assert hasattr(colab, "_launch_registered"), "Register children before releasing execution"
    marker = tmp_path / "must_not_execute"
    store = PrivateStore(tmp_path)
    stopped = []

    def fail_registration(*_):
        raise OSError("Independent registration write failure")

    def confirm_eof_exit(process):
        # Waiting rather than killing proves EOF alone prevents command execution.
        try:
            assert process.wait(timeout=5) == 0 and not marker.exists()
            stopped.append(process.pid)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    monkeypatch.setattr(store, "put", fail_registration)
    monkeypatch.setattr(colab, "_stop", confirm_eof_exit)
    with pytest.raises(OSError, match="registration write failure"):
        colab._launch_registered(
            [sys.executable, "-c", "from pathlib import Path; import sys; "
             "Path(sys.argv[1]).touch()", str(marker)],
            store=store, session_id="fixture", control={"active": True, "children": []},
            start_new_session=True,
        )
    assert len(stopped) == 1 and not marker.exists()


def test_time_only_session_settles_elapsed_and_keeps_usd_unknown(simulated_lifecycle):
    colab, config, store, state = simulated_lifecycle
    config.qwen.gpu_budget_hours = 12
    config.qwen.gpu_hourly_rate_usd = None
    result = colab._run_managed(config, None, 60, 30, {"name": "synthetic GPU"})
    assert result["status"] == "executed"
    receipt = result["gpu_costs"]
    assert receipt["managed_session_seconds"] == 9
    assert receipt["managed_session_cost_usd"] is None
    assert receipt["cumulative_gpu_seconds"] == 9
    assert receipt["remaining_gpu_seconds"] == 43200 - 9
    assert not store.path("budget").exists()


def test_cumulative_remaining_time_caps_session_and_stops_worker(simulated_lifecycle):
    colab, config, store, state = simulated_lifecycle
    config.qwen.gpu_budget_hours = 12
    config.qwen.gpu_hourly_rate_usd = None
    colab.initialize_gpu_budget(config.run_dir, 12, previously_used_seconds=43150)
    state.worker_exit = None
    result = colab._run_managed(config, None, 3600, 30, {"name": "synthetic GPU"})
    assert result["status"] == "partial_or_failed"
    assert store.get("sessions", "synthetic_session")["deadline_epoch"] == 1150
    assert result["gpu_costs"]["cumulative_gpu_seconds"] <= 43200
    assert state.stopped == ["worker", "server", "watchdog"]
    launched = list(state.launched)
    with pytest.raises(ValueError, match="insufficient remaining time"):
        colab._run_managed(config, None, 3600, 30, {"name": "synthetic GPU"})
    assert state.launched == launched


def test_lost_time_only_session_blocks_other_phase_until_reconciled(simulated_lifecycle):
    colab, config, store, state = simulated_lifecycle
    config.qwen.gpu_budget_hours = 12
    config.qwen.gpu_hourly_rate_usd = None
    state.cleanup_failure = True
    with pytest.raises(OSError, match="shutdown uncertainty"):
        colab._run_managed(config, None, 60, 30, {"name": "synthetic GPU"})
    config.run_dir = str(store.root.parent.parent / "next-phase")
    with pytest.raises(ValueError, match="Reconcile"):
        colab._run_managed(config, None, 60, 30, {"name": "synthetic GPU"})
    assert state.launched == ["watchdog", "server", "worker"]
