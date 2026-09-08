"""Short-lived, loopback-only GPU supervision; importing this module starts nothing."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
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
    command = [
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
    if config.structured_summary_mode != "prompt" or config.monitor_output_mode != "prompt":
        command.extend(["--structured-outputs-config.backend", "xgrammar"])
    return command


def _time_receipt(ledger: BudgetLedger) -> dict:
    return dict(
        budget_dir=str(ledger.store.root),
        limit_seconds=ledger.limit,
        committed_seconds=ledger.committed,
        remaining_seconds=max(0.0, ledger.limit - ledger.committed),
        uncertain_sessions=sorted(set(ledger.amounts) - ledger.settled),
    )


def initialize_gpu_budget(
    run_dir: Path, max_gpu_hours: float, *, previously_used_seconds: float = 0,
) -> dict:
    """Pin one cumulative budget for all sibling phase directories, in seconds."""
    from context_audit.runner import run_lock

    if (
        not math.isfinite(max_gpu_hours) or max_gpu_hours <= 0
        or not math.isfinite(previously_used_seconds) or previously_used_seconds < 0
        or previously_used_seconds > max_gpu_hours * 3600
    ):
        raise ValueError("Finite positive GPU hours and valid previously used seconds required")
    budget_dir = Path(run_dir).parent / "gpu_budget"
    with run_lock(budget_dir / "supervisor"):
        return _initialize_gpu_budget(budget_dir, max_gpu_hours, previously_used_seconds)


def _initialize_gpu_budget(budget_dir, hours, prior):
    store = PrivateStore(budget_dir)
    pin = dict(limit_seconds=hours * 3600, previously_used_seconds=prior)
    saved = store.get("manifests", "limit")
    if saved and saved != pin:
        raise ValueError("Cumulative GPU budget or initial debit differs from its saved value")
    if saved is None:
        store.put("manifests", "limit", pin)
    ledger = BudgetLedger(store, pin["limit_seconds"], unit="seconds")
    if prior:
        key = "previously_used_gpu_time"
        if key not in ledger.amounts:
            ledger.reserve(key, prior)
        if key not in ledger.settled:
            ledger.settle(key, prior)
    return _time_receipt(ledger)


def _gpu_time_ledger(run_dir, hours):
    if hours is None:
        return None
    store = PrivateStore(Path(run_dir).parent / "gpu_budget")
    saved = store.get("manifests", "limit")
    if saved is None:
        _initialize_gpu_budget(store.root, hours, 0)
    elif saved["limit_seconds"] != hours * 3600:
        raise ValueError("Cumulative GPU budget differs from its saved value")
    else:
        _initialize_gpu_budget(store.root, hours, saved["previously_used_seconds"])
    return BudgetLedger(store, hours * 3600, unit="seconds")


def record_external_gpu_time(
    run_dir: Path,
    max_gpu_hours: float,
    *,
    usage_id: str,
    elapsed_seconds: float,
    confirmed: bool = False,
) -> dict:
    """Debit verified setup/idle allocation once, without replenishing the shared cap.

    Each distinct allocation period needs its own stable user-supplied identity.
    This is reported past use, so an overrun is preserved before stopping the run.
    """
    from context_audit.runner import run_lock

    if (
        confirmed is not True
        or type(max_gpu_hours) not in (int, float)
        or not math.isfinite(max_gpu_hours) or max_gpu_hours <= 0
        or type(elapsed_seconds) not in (int, float)
        or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0
        or not isinstance(usage_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", usage_id) is None
    ):
        raise ValueError("Confirm finite external GPU elapsed seconds and a safe stable usage ID")
    budget_dir = Path(run_dir).parent / "gpu_budget"
    key = f"external-{usage_id}"
    with run_lock(budget_dir / "supervisor"):
        ledger = _gpu_time_ledger(run_dir, max_gpu_hours)
        saved = ledger.store.get("external_usage", key)
        if saved is not None and saved["elapsed_seconds"] != elapsed_seconds:
            raise ValueError("External GPU usage differs from the saved debit for this ID")
        if key in ledger.settled:
            if ledger.amounts[key] != elapsed_seconds:
                raise ValueError("External GPU usage differs from its settled ledger amount")
            return _time_receipt(ledger)
        if saved is None:
            ledger.store.put("external_usage", key, dict(
                usage_id=usage_id, elapsed_seconds=elapsed_seconds,
                confirmed_by_user=True, at=utc_now(),
            ))
        if key not in ledger.amounts:
            ledger.reserve(key, min(elapsed_seconds, max(0.0, ledger.limit - ledger.committed)))
        # Settlement journals the actual reported use even if it exceeds the cap,
        # then raises; later budget access also fails instead of granting more time.
        ledger.settle(key, elapsed_seconds)
        return _time_receipt(ledger)


def require_managed_session(config: AuditConfig, max_cost_usd: float | None) -> None:
    session_id = os.environ.get("CONTEXT_AUDIT_GPU_SESSION", "")
    if not session_id:
        raise ValueError("Qwen live runs require the managed Colab supervisor")
    store = PrivateStore(Path(config.run_dir) / "gpu_sessions")
    session = store.get("sessions", session_id)
    ledgers = []
    if max_cost_usd is not None:
        ledgers.append(BudgetLedger(store, max_cost_usd))
    time_ledger = _gpu_time_ledger(config.run_dir, config.qwen.gpu_budget_hours)
    if time_ledger is not None:
        ledgers.append(time_ledger)
    if (
        not session or not ledgers
        or any(session_id not in ledger.amounts or session_id in ledger.settled
               for ledger in ledgers)
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
    max_cost_usd: float | None = None,
    confirmed: bool = False,
) -> None:
    """Record user-verified elapsed runtime after a lost supervisor/VM receipt."""
    from context_audit.runner import run_lock

    if not confirmed or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError(
            "Explicitly confirm the observed GPU elapsed seconds before reconciliation"
        )
    run_dir = Path(run_dir)
    store = PrivateStore(run_dir / "gpu_sessions")
    session = store.get("sessions", session_id)
    if not session:
        raise ValueError("Session is absent or already settled")
    if session.get("deadline_epoch", 0) > time.time():
        raise ValueError("Wait for the active session deadline before reconciliation")
    hours = session.get("gpu_budget_hours")
    lock_dir = run_dir.parent / "gpu_budget" if hours is not None else run_dir
    with run_lock(lock_dir / "supervisor"):
        ledgers = []
        if max_cost_usd is not None:
            ledgers.append((BudgetLedger(store, max_cost_usd),
                            elapsed_seconds * session["hourly_rate_usd"] / 3600))
        time_ledger = _gpu_time_ledger(run_dir, hours)
        if time_ledger is not None:
            ledgers.append((time_ledger, elapsed_seconds))
        if not ledgers or any(session_id not in ledger.amounts for ledger, _ in ledgers):
            raise ValueError("Session is absent or already settled")
        if all(session_id in ledger.settled for ledger, _ in ledgers):
            raise ValueError("Session is absent or already settled")
        for ledger, amount in ledgers:
            if session_id not in ledger.settled:
                ledger.settle(session_id, amount)
        store.put("sessions", session_id, dict(
            **session, reconciliation=dict(
                elapsed_seconds=elapsed_seconds, confirmed_by_user=True, at=utc_now(),
            ),
        ))
        finalize_gpu_costs(run_dir, max_cost_usd)


def _dollar_ledger_seconds(ledger: BudgetLedger | None, sessions: dict) -> float | None:
    """Recover measured or conservative time from older USD-only session journals."""
    if ledger is None:
        return None
    seconds = 0.0
    for key, amount in ledger.amounts.items():
        session = sessions.get(key, {})
        measured = session.get("elapsed_seconds", session.get("reconciliation", {}).get(
            "elapsed_seconds"
        ))
        if key in ledger.settled and measured is not None:
            seconds += measured
        else:
            rate = session.get("hourly_rate_usd")
            if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
                return None
            seconds += amount * 3600 / rate
    return seconds


def finalize_gpu_costs(run_dir: Path, max_cost_usd: float | None = None) -> dict:
    """Report elapsed GPU time and optional estimated USD; unknown USD stays null."""
    import pandas as pd

    from context_audit.metrics import validate_rows
    from context_audit.runner import write_scores

    run_dir = Path(run_dir)
    store = PrivateStore(run_dir)
    session_store = PrivateStore(run_dir / "gpu_sessions")
    sessions = {path.stem: json.loads(path.read_text())
                for path in (session_store.root / "sessions").glob("*.json")}
    usd_ledger = BudgetLedger(session_store, max_cost_usd) if max_cost_usd is not None else None
    hours = next((s["gpu_budget_hours"] for s in sessions.values()
                  if s.get("gpu_budget_hours") is not None), None)
    time_ledger = _gpu_time_ledger(run_dir, hours)
    if time_ledger is not None:
        amounts = {key: time_ledger.amounts[key] for key in sessions if key in time_ledger.amounts}
        uncertain = bool(set(amounts) - time_ledger.settled)
        seconds = sum(amounts.values())
        cost = (sum(amount * sessions[key]["hourly_rate_usd"] / 3600
                    for key, amount in amounts.items())
                if amounts and all(sessions[key]["hourly_rate_usd"] is not None
                                   for key in amounts) else None)
    else:
        uncertain = bool(set(usd_ledger.amounts) - usd_ledger.settled) if usd_ledger else False
        seconds = _dollar_ledger_seconds(usd_ledger, sessions)
        cost = usd_ledger.committed if usd_ledger else None
    if usd_ledger is not None:
        cost = usd_ledger.committed
        uncertain |= bool(set(usd_ledger.amounts) - usd_ledger.settled)
    receipt = dict(
        managed_session_cost_usd=cost,
        managed_session_seconds=seconds,
        cost_is_upper_bound=uncertain,
        time_is_upper_bound=uncertain,
        sessions=len(sessions) if sessions else len(usd_ledger.amounts) if usd_ledger else 0,
        allocation="Session cost minus attributed request costs, equally across observed units",
        scope="Managed server startup through teardown; excludes GPU allocation outside this block",
        price_basis=("User-supplied effective hourly rate; not a provider invoice"
                     if cost is not None else "Unknown; no GPU hourly rate supplied"),
        at=utc_now(),
    )
    if time_ledger is not None:
        receipt.update(
            cumulative_gpu_seconds=time_ledger.committed,
            remaining_gpu_seconds=max(0.0, time_ledger.limit - time_ledger.committed),
            gpu_limit_seconds=time_ledger.limit,
        )
    score_path = run_dir / "scores.csv"
    if score_path.exists():
        frame = validate_rows(pd.read_csv(score_path), allow_partial_pairs=True)
        if not frame.empty:
            request_cost = frame.summary_cost_usd + frame.monitor_cost_usd
            priced = cost is not None and request_cost.notna().all()
            overhead = max(0.0, cost - float(request_cost.sum())) if priced else None
            frame["infrastructure_cost_usd"] = overhead / len(frame) if priced else None
            frame["cost_usd"] = request_cost + frame.infrastructure_cost_usd if priced else None
            if not priced:
                frame["summary_cost_usd"] = None
                frame["monitor_cost_usd"] = None
            frame["cost_is_upper_bound"] |= uncertain
            write_scores(score_path, frame.to_dict("records"))
            receipt.update(
                attributed_request_cost_usd=float(request_cost.sum()) if priced else None,
                infrastructure_cost_usd=overhead,
                reported_total_cost_usd=float(frame.cost_usd.sum()) if priced else None,
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
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    cards = [line.split(",") for line in result.stdout.strip().splitlines()]
    if len(cards) != 1 or len(cards[0]) != 4:
        raise ValueError("Expected one NVIDIA GPU with reported memory, driver and capability")
    name, memory, driver, capability = (value.strip() for value in cards[0])
    try:
        memory_mib = float(memory)
        compute = tuple(int(value) for value in capability.split("."))
    except ValueError as exc:
        raise ValueError("Could not verify GPU memory or compute capability") from exc
    if (
        not name or not driver or not math.isfinite(memory_mib) or memory_mib < 75000
        or len(compute) != 2 or any(value < 0 for value in compute) or compute < (8, 0)
    ):
        raise ValueError(
            "One GPU with at least 75,000 MiB VRAM and native BF16 capability >= 8.0 required"
        )
    # Hardware eligibility does not attest installed CUDA/vLLM kernel compatibility.
    return dict(
        name=name, memory_mib=memory_mib, driver=driver, compute_capability=capability
    )


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
    max_cost_usd: float | None = None,
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
    from context_audit.runner import validate_run_budget

    validate_run_budget(config, max_cost_usd)
    for value in (session_max_seconds, startup_timeout_seconds):
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
    if config.structured_summary_mode != "prompt" or config.monitor_output_mode != "prompt":
        from context_audit.structured_backend import check_structured_backend

        hardware["structured_outputs"] = check_structured_backend()
    url = urlsplit(config.qwen.base_url)
    # Never accidentally contact a server from another notebook/run.
    try:
        with socket.create_connection((url.hostname, url.port), timeout=1):
            raise ValueError("Qwen port already in use; stop the previous owned server first")
    except (ConnectionRefusedError, TimeoutError):
        pass
    lock_dir = (run_dir.parent / "gpu_budget"
                if config.qwen.gpu_budget_hours is not None else run_dir)
    with run_lock(lock_dir / "supervisor"):
        return _run_managed(
            config, max_cost_usd, session_max_seconds, startup_timeout_seconds, hardware
        )


def _run_managed(config, cap, maximum_seconds, startup_seconds, hardware):
    run_dir = Path(config.run_dir)
    store = PrivateStore(run_dir / "gpu_sessions")
    ledger = BudgetLedger(store, cap) if cap is not None else None
    time_ledger = _gpu_time_ledger(run_dir, config.qwen.gpu_budget_hours)
    ledgers = [item for item in (ledger, time_ledger) if item is not None]
    if not ledgers:
        raise ValueError("A finite GPU time or monetary budget is required")
    if any(set(item.amounts) - item.settled for item in ledgers):
        raise ValueError("Reconcile the previous uncertain GPU session before allocating another")
    rate = config.qwen.gpu_hourly_rate_usd
    seconds = maximum_seconds
    if ledger is not None:
        seconds = min(seconds, (cap - ledger.committed) * 3600 / rate)
    if time_ledger is not None:
        seconds = min(seconds, time_ledger.limit - time_ledger.committed)
    if seconds <= 15:
        raise ValueError("GPU budget has insufficient remaining time including shutdown allowance")
    session_id = uuid.uuid4().hex
    if ledger is not None:
        ledger.reserve(session_id, seconds * rate / 3600)
    if time_ledger is not None:
        time_ledger.reserve(session_id, seconds)
    started, deadline = time.monotonic(), time.time() + seconds
    # vLLM 0.28.0 supports this switch before import. Use the native sampler for
    # every phase, including dummy profiling: FlashInfer's JIT architecture check
    # failed on the author's SM120 runtime after the checkpoint loaded successfully.
    engine_environment = dict(
        VLLM_USE_FLASHINFER_SAMPLER="0",
        VLLM_NO_USAGE_STATS="1",
        HF_HUB_DISABLE_TELEMETRY="1",
    )
    metadata = dict(
        at=utc_now(),
        deadline_epoch=deadline,
        hourly_rate_usd=rate,
        gpu_budget_hours=config.qwen.gpu_budget_hours,
        gpu_budget_dir=str(time_ledger.store.root) if time_ledger is not None else None,
        qwen_config=config.qwen.model_dump(),
        hardware=hardware,
        python=sys.version.split()[0],
        engine_environment=engine_environment,
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
                env=dict(os.environ, **engine_environment),
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
                    ] + (["--max-cost-usd", str(cap)] if cap is not None else []),
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
        if time_ledger is not None:
            time_ledger.settle(session_id, elapsed)
        if ledger is not None:
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
