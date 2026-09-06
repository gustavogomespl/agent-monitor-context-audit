"""Short-lived, loopback-only GPU supervision; importing this module starts nothing."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import yaml

from context_audit.provider import BudgetLedger, utc_now
from context_audit.runtime_models import AuditConfig
from context_audit.storage import PrivateStore


def vllm_command(config: AuditConfig) -> list[str]:
    qwen = config.qwen
    qwen.validate_live()
    url = urlsplit(qwen.base_url)
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        config.monitor_model,
        "--revision",
        qwen.model_revision,
        "--tokenizer-revision",
        qwen.model_revision,
        "--served-model-name",
        config.monitor_model,
        "--host",
        url.hostname,
        "--port",
        str(url.port),
        "--dtype",
        qwen.dtype,
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(qwen.max_model_len),
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        str(qwen.gpu_memory_utilization),
        "--max-num-batched-tokens",
        str(qwen.max_num_batched_tokens),
        "--generation-config",
        "vllm",
        "--seed",
        str(qwen.seed),
        "--language-model-only",
        "--enforce-eager",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--no-enable-log-requests",
        "--default-chat-template-kwargs",
        '{"enable_thinking":false,"preserve_thinking":false}',
    ]


def require_managed_session(config: AuditConfig, max_cost_usd: float) -> None:
    session_id = os.environ.get("CONTEXT_AUDIT_GPU_SESSION", "")
    if not session_id:
        raise ValueError("Qwen live runs require the managed Colab supervisor")
    store = PrivateStore(Path(config.run_dir) / "gpu_sessions")
    session = store.get("sessions", session_id)
    ledger = BudgetLedger(store, max_cost_usd)
    if (
        not session
        or session_id not in ledger.amounts
        or session_id in ledger.settled
        or session["deadline_epoch"] <= time.time()
        or session["hourly_rate_usd"] != config.qwen.gpu_hourly_rate_usd
        or session["qwen_config"] != config.qwen.model_dump()
    ):
        raise ValueError("No active matching managed GPU session reservation")


def reconcile_gpu_session(
    run_dir: Path,
    session_id: str,
    *,
    elapsed_seconds: float,
    max_cost_usd: float,
    confirmed: bool = False,
) -> None:
    """Record user-verified elapsed runtime after a lost supervisor/VM receipt."""
    if not confirmed or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError(
            "Explicitly confirm the observed GPU elapsed seconds before reconciliation"
        )
    store = PrivateStore(Path(run_dir) / "gpu_sessions")
    ledger = BudgetLedger(store, max_cost_usd)
    session = store.get("sessions", session_id)
    if not session or session_id not in ledger.amounts or session_id in ledger.settled:
        raise ValueError("Session is absent or already settled")
    ledger.settle(session_id, elapsed_seconds * session["hourly_rate_usd"] / 3600)
    store.put(
        "sessions",
        session_id,
        dict(
            **session,
            reconciliation=dict(
                elapsed_seconds=elapsed_seconds,
                confirmed_by_user=True,
                at=utc_now(),
            ),
        ),
    )
    finalize_gpu_costs(Path(run_dir), max_cost_usd)


def finalize_gpu_costs(run_dir: Path, max_cost_usd: float) -> dict:
    """Allocate measured session overhead equally across observed evaluation units."""
    import pandas as pd

    from context_audit.metrics import validate_rows
    from context_audit.runner import write_scores

    store = PrivateStore(run_dir)
    ledger = BudgetLedger(PrivateStore(run_dir / "gpu_sessions"), max_cost_usd)
    uncertain = bool(set(ledger.amounts) - ledger.settled)
    receipt = dict(
        managed_session_cost_usd=ledger.committed,
        cost_is_upper_bound=uncertain,
        sessions=len(ledger.amounts),
        allocation="Session cost minus attributed request costs, equally across observed units",
        scope="Managed server startup through teardown; excludes GPU allocation outside this block",
        price_basis="User-supplied effective hourly rate; not a provider invoice",
        at=utc_now(),
    )
    score_path = run_dir / "scores.csv"
    if score_path.exists():
        frame = validate_rows(pd.read_csv(score_path), allow_partial_pairs=True)
        if not frame.empty:
            request_cost = frame.summary_cost_usd + frame.monitor_cost_usd
            overhead = max(0.0, ledger.committed - float(request_cost.sum()))
            frame["infrastructure_cost_usd"] = overhead / len(frame)
            frame["cost_usd"] = request_cost + frame.infrastructure_cost_usd
            frame["cost_is_upper_bound"] |= uncertain
            write_scores(score_path, frame.to_dict("records"))
            receipt.update(
                attributed_request_cost_usd=float(request_cost.sum()),
                infrastructure_cost_usd=overhead,
                reported_total_cost_usd=float(frame.cost_usd.sum()),
                observed_units=len(frame),
            )
            completion = store.get("manifests", "completion")
            if completion:
                completion["scores_sha256"] = hashlib.sha256(score_path.read_bytes()).hexdigest()
                completion["gpu_costs"] = receipt
                store.put("manifests", "completion", completion)
    store.put("manifests", "gpu_costs", receipt)
    return receipt


def _stop(process) -> None:
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    # An exited leader can leave EngineCore descendants alive. Kill the owned
    # group even after wait() reports that the leader completed successfully.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=3)
    time.sleep(0.1)


def _hardware() -> dict:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    cards = [line.split(",") for line in result.stdout.strip().splitlines()]
    if len(cards) != 1 or "H100" not in cards[0][0] or float(cards[0][1]) < 75000:
        raise ValueError("This notebook requires one H100 with at least 75,000 MiB visible VRAM")
    return dict(name=cards[0][0].strip(), memory_mib=float(cards[0][1]), driver=cards[0][2].strip())


# This independent watchdog outlives a notebook KeyboardInterrupt or dead kernel.
# Its only targets are process groups explicitly started by this supervisor.
_WATCHDOG = """
import json, os, signal, sys, time
path, deadline, owner = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
while True:
    try:
        state = json.load(open(path))
        if not state['active']: break
        try: os.kill(owner, 0); alive = True
        except ProcessLookupError: alive = False
        if time.time() >= deadline or not alive:
            for pid in state['children']:
                try: os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError: pass
            break
    except (FileNotFoundError, ValueError): pass
    time.sleep(0.25)
"""


_LAUNCH_BARRIER = """
import os, sys
fd = int(sys.argv[1])
release = os.read(fd, 1)
os.close(fd)
if release == b'1':
    os.execvpe(sys.argv[2], sys.argv[2:], os.environ)
"""


def _launch_registered(command, *, store, session_id, control, **popen_kwargs):
    """Register the owned process group before allowing its command to execute.

    An abruptly lost parent closes the only pipe writer. The waiting child then
    exits on EOF, including when its PID never reached the watchdog's journal.
    exec preserves the registered PID, process group, environment and output FDs.
    """
    read_fd, write_fd = os.pipe()
    process = None
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", _LAUNCH_BARRIER, str(read_fd), *command],
            pass_fds=(read_fd,),
            **popen_kwargs,
        )
        os.close(read_fd)
        read_fd = None
        control["children"].append(process.pid)
        store.put("watchdogs", session_id, control)
        if os.write(write_fd, b"1") != 1:
            raise OSError("Child execution barrier was not released")
        return process
    except BaseException:
        # Close before waiting so a registration failure lets the child exit on
        # EOF without importing vLLM or starting the experiment worker.
        os.close(write_fd)
        write_fd = None
        _stop(process)
        raise
    finally:
        if read_fd is not None:
            os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)


def run_colab_experiment(
    config: AuditConfig,
    *,
    max_cost_usd: float,
    session_max_seconds: float = 3600,
    startup_timeout_seconds: float = 900,
) -> dict:
    """Explicitly run one managed stage; never invoked on notebook open/import."""
    from context_audit.dataset import load_dataset
    from context_audit.runner import require_private_path, run_lock

    if (
        config.provider != "qwen_local"
        or not config.data_use_confirmed
        or not config.rubric_reviewed
    ):
        raise ValueError("Select Qwen, confirm data use and review the rubric before GPU execution")
    config.qwen.validate_live()
    for value in (max_cost_usd, session_max_seconds, startup_timeout_seconds):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Explicit positive finite financial and session limits required")
    run_dir = Path(config.run_dir)
    require_private_path(run_dir, Path("runs/private"))
    require_private_path(Path(config.dataset_dir), Path("data/private"))
    inputs, _ = load_dataset(Path(config.dataset_dir), config.split)
    if not inputs or any(t.data_origin != "sleight_bench" for t in inputs):
        raise ValueError("Real inventoried data required; never replace it with fixtures")
    actual_version = importlib.metadata.version("vllm")
    if actual_version != config.qwen.vllm_version:
        raise ValueError("Installed vLLM differs from the pinned configuration")
    runtime_versions = {
        name: importlib.metadata.version(name)
        for name in ("vllm", "torch", "transformers", "tokenizers", "triton", "safetensors")
    }
    if config.qwen.runtime_versions != runtime_versions:
        raise ValueError("Inference packages differ from the saved runtime pin; rerun Colab setup")
    hardware = _hardware()
    url = urlsplit(config.qwen.base_url)
    # Never accidentally contact a server from another notebook/run.
    try:
        with socket.create_connection((url.hostname, url.port), timeout=1):
            raise ValueError("Qwen port already in use; stop the previous owned server first")
    except (ConnectionRefusedError, TimeoutError):
        pass
    with run_lock(run_dir / "supervisor"):
        return _run_managed(
            config, max_cost_usd, session_max_seconds, startup_timeout_seconds, hardware
        )


def _run_managed(config, cap, maximum_seconds, startup_seconds, hardware):
    run_dir = Path(config.run_dir)
    store = PrivateStore(run_dir / "gpu_sessions")
    ledger = BudgetLedger(store, cap)
    if set(ledger.amounts) - ledger.settled:
        raise ValueError("Reconcile the previous uncertain GPU session before allocating another")
    rate = config.qwen.gpu_hourly_rate_usd
    seconds = min(maximum_seconds, (cap - ledger.committed) * 3600 / rate)
    if seconds <= 15:
        raise ValueError("GPU budget has insufficient remaining time including shutdown allowance")
    session_id = uuid.uuid4().hex
    ledger.reserve(session_id, seconds * rate / 3600)
    started, deadline = time.monotonic(), time.time() + seconds
    metadata = dict(
        at=utc_now(),
        deadline_epoch=deadline,
        hourly_rate_usd=rate,
        qwen_config=config.qwen.model_dump(),
        hardware=hardware,
        python=sys.version.split()[0],
    )
    store.put("sessions", session_id, metadata)
    command = vllm_command(config)
    config_file = store.path("configs", session_id)
    config_file.write_text(yaml.safe_dump(config.model_dump()))
    control = dict(active=True, children=[])
    store.put("watchdogs", session_id, control)
    server = worker = watchdog = None
    outcome = "partial_or_failed"
    exit_code = None
    try:
        watchdog = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _WATCHDOG,
                str(store.path("watchdogs", session_id)),
                str(deadline - 8),
                str(os.getpid()),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        with store.path("server_logs", session_id).open("w") as server_log:
            server = _launch_registered(
                command,
                store=store,
                session_id=session_id,
                control=control,
                stdout=server_log,
                stderr=server_log,
                start_new_session=True,
                env=dict(os.environ, VLLM_NO_USAGE_STATS="1", HF_HUB_DISABLE_TELEMETRY="1"),
            )
            ready_until = min(time.monotonic() + startup_seconds, started + seconds - 10)
            with httpx.Client(trust_env=False, follow_redirects=False, timeout=1) as client:
                while True:
                    if server.poll() is not None:
                        raise ValueError("vLLM startup failed; inspect the private server log")
                    if time.monotonic() >= ready_until:
                        raise ValueError(
                            "vLLM startup exceeded the configured time or financial cap"
                        )
                    try:
                        if client.get(config.qwen.base_url + "/health").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.5)
            env = dict(os.environ, CONTEXT_AUDIT_GPU_SESSION=session_id)
            with store.path("runner_logs", session_id).open("w") as runner_log:
                worker = _launch_registered(
                    [
                        sys.executable,
                        "-m",
                        "context_audit.cli",
                        "run",
                        "--live",
                        "--config",
                        str(config_file),
                        "--max-cost-usd",
                        str(cap),
                    ],
                    store=store,
                    session_id=session_id,
                    control=control,
                    env=env,
                    stdout=runner_log,
                    stderr=runner_log,
                    start_new_session=True,
                )
                while worker.poll() is None and time.monotonic() < started + seconds - 10:
                    time.sleep(0.5)
                exit_code = worker.poll()
                outcome = "executed" if exit_code == 0 else "partial_or_failed"
    finally:
        _stop(worker)
        _stop(server)
        control["active"] = False
        store.put("watchdogs", session_id, control)
        _stop(watchdog)
        elapsed = time.monotonic() - started
        # Known teardown settles actual elapsed time. SIGKILL/VM loss leaves the
        # reservation intact for explicit user reconciliation on the next session.
        ledger.settle(session_id, elapsed * rate / 3600)
        store.put(
            "sessions",
            session_id,
            dict(
                **metadata,
                elapsed_seconds=elapsed,
                finished_at=utc_now(),
                exit_code=exit_code,
            ),
        )
        finalize_gpu_costs(run_dir, cap)
    return dict(
        status=outcome,
        session_id=session_id,
        exit_code=exit_code,
        gpu_costs=PrivateStore(run_dir).get("manifests", "gpu_costs"),
        run_dir=str(run_dir),
    )
