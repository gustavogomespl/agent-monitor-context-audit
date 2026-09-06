"""Label-free model pipeline and resumable, single-process experiment orchestration."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import random
import subprocess
from contextlib import contextmanager
from pathlib import Path

import yaml

from context_audit.provider import (
    AnthropicProvider,
    BudgetLedger,
    TokenCountError,
    load_prices,
    utc_now,
)
from context_audit.render import render_body, render_task_header
from context_audit.representations import (
    CONDITIONS,
    budget_for,
    evidence_ids,
    head_tail,
    validate_summary,
)
from context_audit.runtime_models import (
    AuditConfig,
    MonitorDecision,
    MonitorResult,
    Representation,
    RunManifest,
)
from context_audit.schemas import EvaluationLabel, TranscriptInput
from context_audit.storage import PrivateStore, digest


def load_config(path: Path) -> AuditConfig:
    return AuditConfig.model_validate(yaml.safe_load(path.read_text()))


def prompts(config: AuditConfig) -> dict[str, str]:
    return {
        name: (Path(config.prompt_dir) / f"{name}.txt").read_text()
        for name in ("monitor", "summary_common", "summary_free", "summary_structured")
    }


def build_monitor_prompt(transcript: TranscriptInput, representation_text: str) -> str:
    return render_task_header(transcript) + "\n<history>\n" + representation_text + "\n</history>"


def _identity(transcript, config, condition, repetition, phase, attempt):
    return dict(
        normalized_hash=digest(transcript.model_dump(mode="json")),
        config_hash=digest(config.model_dump()),
        condition=condition,
        repetition=repetition,
        phase=phase,
        attempt=attempt,
    )


def make_representation(
    transcript: TranscriptInput,
    condition: str,
    config: AuditConfig,
    provider,
    store: PrivateStore,
    repetition: int,
    summary_window: int,
) -> tuple[Representation, list[dict]]:
    body = render_body(transcript)

    def count(text):
        return provider.count_text(config.monitor_model, text)

    tokens = count(body)
    budget = budget_for(tokens, config.token_fraction, config.token_minimum, config.token_maximum)
    common = dict(
        transcript_id=transcript.transcript_id,
        condition=condition,
        budget=tokens if condition == "full" else budget,
        body_tokens=tokens,
        data_origin=transcript.data_origin,
        counter_method=provider.counter_method,
        compression_applied=condition != "full" and budget < tokens,
    )
    calls = []
    if condition == "full" or budget == tokens:
        text, status = body, "ok"
    elif condition == "head_tail":
        try:
            text, status = head_tail(body, budget, count), "ok"
        except TokenCountError:
            text, status = "", "api_error"
    elif condition in ("free_summary", "structured_summary"):
        all_prompts = prompts(config)
        system = (
            all_prompts["summary_common"]
            + "\n"
            + all_prompts["summary_free" if condition == "free_summary" else "summary_structured"]
        )
        request = (
            f"Token ceiling: {budget}, counted in the monitor model token unit.\n"
            + build_monitor_prompt(transcript, body)
        )
        visible = {event.event_id for event in transcript.events}
        text, status, keys = "", "invalid_output", []
        for attempt in range(config.max_attempts):
            key, call = provider.generate(
                model=config.summarizer_model,
                system=system,
                text=request,
                max_tokens=config.summary_max_tokens,
                context_window=summary_window,
                identity=_identity(transcript, config, condition, repetition, "summary", attempt),
            )
            calls.append(call)
            keys.append(key)
            status = call["status"]
            if status == "ok":
                try:
                    text = validate_summary(call["text"], condition, budget, count, visible)
                    break
                except TokenCountError:
                    # The summary call is already durable and billable. Do not regenerate it
                    # merely because its output could not be measured by the counter.
                    text, status = "", "api_error"
                    break
                except ValueError as exc:
                    status = (
                        "budget_violation" if count(call["text"]) > budget else "invalid_output"
                    )
                    repair = str(exc)
            else:
                repair = "The previous request failed to produce a complete valid response."
            if status in ("refusal", "context_limit") or call.get("error") in (
                "financial_cap",
                "maximum_generation_calls",
                "uncertain_previous_request",
            ):
                break
            # Regeneration sees original data and a generic format/length error only.
            request = (
                f"Token ceiling: {budget}. Regenerate once: {repair}\n"
                + build_monitor_prompt(transcript, body)
            )
        common["call_keys"] = keys
        if status != "ok":
            text = ""
    else:
        raise ValueError("Unknown condition")
    rep = Representation(
        **common, text=text, status=status, measured_tokens=count(text), content_hash=digest(text)
    )
    key = digest(_identity(transcript, config, condition, repetition, "representation", 0))
    store.put(
        "representations",
        key,
        dict(representation=rep.model_dump(), canaries=getattr(provider, "canaries", [])),
    )
    return rep, calls


def monitor_representation(
    transcript: TranscriptInput,
    representation: Representation,
    config: AuditConfig,
    provider,
    repetition: int,
    monitor_window: int,
) -> tuple[MonitorResult, list[dict]]:
    system = prompts(config)["monitor"]
    base = dict(
        model_requested=config.monitor_model,
        prompt_hash=digest(system),
        representation_hash=representation.content_hash,
    )
    if representation.status != "ok":
        return MonitorResult(status=representation.status, **base), []
    request = build_monitor_prompt(transcript, representation.text)
    visible = evidence_ids(representation.text) & {event.event_id for event in transcript.events}
    calls, keys, status, decision = [], [], "invalid_output", None
    for attempt in range(config.max_attempts):
        key, call = provider.generate(
            model=config.monitor_model,
            system=system,
            text=request,
            max_tokens=config.monitor_max_tokens,
            context_window=monitor_window,
            identity=_identity(
                transcript, config, representation.condition, repetition, "monitor", attempt
            ),
        )
        calls.append(call)
        keys.append(key)
        status = call["status"]
        if status == "ok":
            try:
                decision = MonitorDecision.model_validate_json(call["text"])
                if not set(decision.evidence_event_ids) <= visible:
                    raise ValueError("Monitor cited evidence it could not see")
                break
            except ValueError:
                decision, status = None, "invalid_output"
        if status in ("refusal", "context_limit", "budget_violation") or call.get("error") == (
            "uncertain_previous_request"
        ):
            break
        # No score or another condition's decision is fed into the repair request.
        request = (
            "Return exactly the requested JSON schema with numeric score and only "
            "visible event IDs. This is the one allowed format regeneration.\n"
            + build_monitor_prompt(transcript, representation.text)
        )
    usage = {
        k: sum(c.get("usage", {}).get(k, 0) for c in calls)
        for k in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
    }
    result = MonitorResult(
        **base,
        status=status,
        **(decision.model_dump() if decision else {}),
        model_returned=calls[-1].get("model") if calls else None,
        call_keys=keys,
        usage=usage,
        latency_seconds=sum(c.get("latency_seconds", 0) for c in calls),
        cost_usd=sum(c.get("cost_usd", 0) for c in calls),
    )
    return result, calls


def code_state() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
    ).stdout.strip()
    files = sorted([*Path("src").rglob("*.py"), Path("pyproject.toml"), Path("uv.lock")])
    return dict(commit=commit, hash=digest({str(p): p.read_text() for p in files if p.exists()}))


def protocol_signature(config: AuditConfig, dataset_manifest: dict) -> dict:
    values = config.model_dump()
    for key in ("run_dir", "protocol_file"):
        values.pop(key)
    return dict(
        config=values,
        dataset_manifest_hash=digest(dataset_manifest),
        prompts={k: digest(v) for k, v in prompts(config).items()},
        code_hash=code_state()["hash"],
    )


@contextmanager
def run_lock(path: Path):
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (path / ".run.lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another process is actively using this run directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _model_window(info: dict, configured: int | None, output: int) -> int:
    available = info.get("max_input_tokens")
    if configured is None:
        if not isinstance(available, int) or available <= 0:
            raise ValueError("Model endpoint has no context limit; configure a documented limit")
        configured = available
    if available and configured > available:
        raise ValueError("Configured context window exceeds model capability")
    if info.get("max_tokens") and output > info["max_tokens"]:
        raise ValueError("Configured output maximum exceeds model capability")
    return configured


def _public_row(transcript, label, rep, result, summary_calls, monitor_calls, config, repetition):
    calls = summary_calls + monitor_calls
    return dict(
        transcript_id=transcript.transcript_id,
        scenario_id=label.scenario_id,
        family_id=label.family_id,
        split=label.split,
        label=label.label,
        condition=rep.condition,
        repetition=repetition,
        data_origin=transcript.data_origin,
        status=result.status,
        suspicion_score=result.suspicion_score,
        escalate=result.escalate,
        body_tokens=rep.body_tokens,
        representation_tokens=rep.measured_tokens,
        **{
            k: sum(c.get("usage", {}).get(k, 0) for c in calls)
            for k in (
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
            )
        },
        summary_cost_usd=sum(c.get("cost_usd", 0) for c in summary_calls),
        monitor_cost_usd=result.cost_usd,
        cost_usd=sum(c.get("cost_usd", 0) for c in calls),
        latency_seconds=sum(c.get("latency_seconds", 0) for c in calls),
        cost_is_upper_bound=any(c.get("cost_is_upper_bound", False) for c in calls),
        monitor_model=config.monitor_model,
        summarizer_model=config.summarizer_model,
    )


def write_scores(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def require_private_path(path: Path, root: Path) -> None:
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Content-bearing path must remain inside {root}")
    ignored = subprocess.run(["git", "check-ignore", "--quiet", str(path)], check=False)
    if ignored.returncode != 0:
        raise ValueError("Private output path must be ignored by Git")


def run_experiment(
    config: AuditConfig,
    transcripts: list[TranscriptInput],
    labels: list[EvaluationLabel],
    dataset_manifest: dict,
    *,
    max_cost_usd: float,
    canaries: list[str],
) -> dict:
    require_private_path(Path(config.run_dir), Path("runs/private"))
    require_private_path(Path(config.dataset_dir), Path("data/private"))
    if not config.data_use_confirmed or not config.rubric_reviewed:
        raise ValueError(
            "Confirm API data-use terms and review the fixed rubric before live execution"
        )
    if not config.monitor_model or not config.summarizer_model:
        raise ValueError("Explicit valid monitor_model and summarizer_model IDs are required")
    if not transcripts or any(t.data_origin != "sleight_bench" for t in transcripts):
        raise ValueError("Live experiment requires real, independently inventoried transcripts")
    if not canaries:
        raise ValueError("Source canaries must accompany private derived artifacts")
    if len({x.transcript_id for x in transcripts}) != len(transcripts):
        raise ValueError("Duplicate transcript inputs")
    if config.split_seed != dataset_manifest.get("split_seed"):
        raise ValueError("Configured split seed disagrees with the inventoried split")
    by_label = {label.transcript_id: label for label in labels}
    if set(by_label) != {t.transcript_id for t in transcripts} or len(by_label) != len(labels):
        raise ValueError("Inputs and evaluator labels must be exactly aligned")
    if any(label.split != config.split for label in labels):
        raise ValueError("Split mismatch")
    scenarios = sorted({label.scenario_id for label in labels})
    for scenario in scenarios:
        if sorted(x.label for x in labels if x.scenario_id == scenario) != [0, 1]:
            raise ValueError("Every scenario must contain exactly a benign/malicious pair")
    if config.pilot_pairs:
        chooser = random.Random(config.split_seed)
        chooser.shuffle(scenarios)
        selected = set(scenarios[: config.pilot_pairs])
        transcripts = [t for t in transcripts if by_label[t.transcript_id].scenario_id in selected]
    signature = protocol_signature(config, dataset_manifest)
    if config.split == "test":
        frozen_path = Path(config.protocol_file)
        if (
            not frozen_path.exists()
            or json.loads(frozen_path.read_text())["signature"] != signature
        ):
            raise ValueError("Test scoring requires an unchanged frozen protocol-v1 signature")
        tag = subprocess.run(
            ["git", "rev-parse", "protocol-v1^{commit}"], capture_output=True, text=True
        )
        if tag.returncode or tag.stdout.strip() != code_state()["commit"]:
            raise ValueError("Test scoring requires HEAD at the reviewed protocol-v1 commit/tag")
    prices, price_snapshot = load_prices(
        Path(config.prices_file), [config.monitor_model, config.summarizer_model]
    )
    with run_lock(Path(config.run_dir)):
        return _run_locked(
            config,
            transcripts,
            by_label,
            dataset_manifest,
            signature,
            max_cost_usd,
            canaries,
            prices,
            price_snapshot,
        )


def _run_locked(
    config,
    transcripts,
    labels,
    dataset_manifest,
    signature,
    max_cost_usd,
    canaries,
    prices,
    price_snapshot,
):
    store = PrivateStore(Path(config.run_dir))
    ledger = BudgetLedger(store, max_cost_usd)
    provider = AnthropicProvider(
        store,
        ledger,
        prices,
        timeout=config.timeout_seconds,
        max_calls=config.max_calls,
        canaries=canaries,
    )
    models = {
        name: provider.model_info(name) for name in {config.monitor_model, config.summarizer_model}
    }
    if any(info["id"] != requested for requested, info in models.items()):
        raise ValueError(
            "Pin resolved model IDs from the model endpoint; aliases are not frozen IDs"
        )
    monitor_window = _model_window(
        models[config.monitor_model], config.monitor_context_window, config.monitor_max_tokens
    )
    summary_window = _model_window(
        models[config.summarizer_model], config.summarizer_context_window, config.summary_max_tokens
    )
    schedule = [
        dict(transcript_id=t.transcript_id, condition=c, repetition=r)
        for t in transcripts
        for c in CONDITIONS
        for r in range(config.repetitions)
    ]
    random.Random(config.call_order_seed).shuffle(schedule)
    run_id = digest(dict(signature=signature, schedule=schedule, prices=price_snapshot))
    manifest = RunManifest(
        run_id=run_id,
        dataset_commit=dataset_manifest["dataset_commit"],
        code_commit=code_state()["commit"],
        code_hash=signature["code_hash"],
        config=config.model_dump(),
        seeds=dict(
            split=config.split_seed, calls=config.call_order_seed, bootstrap=config.bootstrap_seed
        ),
        prompt_hashes=signature["prompts"],
        models=models,
        failure_policy="null score; escalate; at most two attempts; uncertain costs reserved",
        created_at=utc_now(),
        prices=price_snapshot,
        data_origin="sleight_bench",
        planned_transcripts=[t.transcript_id for t in transcripts],
        planned_calls=schedule,
        dataset_manifest_hash=digest(dataset_manifest),
        counter_method=provider.counter_method,
    )
    previous = store.get("manifests", "run")
    if previous and previous["run_id"] != run_id:
        raise ValueError("Run directory has a different protocol; use a new explicit run directory")
    if not previous:
        store.put("manifests", "run", manifest.model_dump())
    # Inventory ALL full monitor and summarizer requests before any score generation.
    preflight, problems = [], []
    for t in transcripts:
        body = render_body(t)
        # Establish body counts for every input before any paid generation. Later
        # representation creation can use these cached measurements without guessing.
        body_tokens = provider.count_text(config.monitor_model, body)
        monitor_count = provider.count_request(
            config.monitor_model, prompts(config)["monitor"], build_monitor_prompt(t, body)
        )
        summary_counts = [
            provider.count_request(
                config.summarizer_model,
                prompts(config)["summary_common"] + "\n" + prompts(config)[fmt],
                "Token ceiling: 1024.\n" + build_monitor_prompt(t, body),
            )
            for fmt in ("summary_free", "summary_structured")
        ]
        item = dict(
            transcript_id=t.transcript_id,
            body_tokens=body_tokens,
            full_input_tokens=monitor_count,
            summary_input_tokens=max(summary_counts),
            monitor_window=monitor_window,
            summary_window=summary_window,
        )
        preflight.append(item)
        if monitor_count + config.monitor_max_tokens > monitor_window or (
            max(summary_counts) + config.summary_max_tokens > summary_window
        ):
            problems.append(t.transcript_id)
    store.put("manifests", "preflight", dict(items=preflight, context_limit_ids=problems))
    if problems:
        raise ValueError(f"{len(problems)} transcripts exceed context; revise scope before scoring")
    by_id = {t.transcript_id: t for t in transcripts}
    rows = []
    for item in schedule:
        t = by_id[item["transcript_id"]]
        rep, summary_calls = make_representation(
            t, item["condition"], config, provider, store, item["repetition"], summary_window
        )
        result, monitor_calls = monitor_representation(
            t, rep, config, provider, item["repetition"], monitor_window
        )
        store.put(
            "results",
            digest(item),
            dict(result=result.model_dump(), representation=rep.model_dump(), canaries=canaries),
        )
        row = _public_row(
            t,
            labels[t.transcript_id],
            rep,
            result,
            summary_calls,
            monitor_calls,
            config,
            item["repetition"],
        )
        rows.append(row)
        write_scores(Path(config.run_dir) / "scores.csv", rows)
    state = dict(
        status="executed" if all(r["status"] == "ok" for r in rows) else "partial_or_failed",
        run_id=run_id,
        rows=len(rows),
        expected_rows=len(schedule),
        successful_rows=sum(r["status"] == "ok" for r in rows),
        conservative_committed_usd=ledger.committed,
        costs_include_uncertain_upper_bounds=bool(set(ledger.amounts) - ledger.settled),
        generation_requests=len(ledger.amounts),
        data_origin="sleight_bench",
        scores_sha256=hashlib.sha256(
            (Path(config.run_dir) / "scores.csv").read_bytes()
        ).hexdigest(),
    )
    store.put("manifests", "completion", state)
    return state
