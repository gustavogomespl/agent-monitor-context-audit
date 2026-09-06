"""Small offline-first CLI. Live inference always requires explicit opt-in and budget."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


def demo() -> dict:
    """Four illustrative representations of our own fixture; no model or scoring."""
    from context_audit.dataset import parse_transcript
    from context_audit.render import render_body, render_task_header
    from context_audit.representations import budget_for, head_tail, validate_summary
    from context_audit.storage import PrivateStore

    fixture = Path("tests/fixtures/visible_events.jsonl")
    if not fixture.exists():
        raise ValueError("Run the demo from the repository root (independent fixture required)")
    transcript = parse_transcript(fixture, "t_" + "a" * 24, data_origin="synthetic_fixture")
    body = render_body(transcript)

    def count(text):
        return math.ceil(len(text.encode()) / 4)

    budget = budget_for(count(body))
    free = (
        "The assistant inspected a file [E0001,E0002]. The tool returned content [E0003]. "
        "The user added an instruction [E0004]. The assistant claimed a count [E0005]; "
        "this final claim is distinct from the observed tool output [E0003]."
    )
    structured = json.dumps(
        dict(
            environment_and_state=[],
            actions_and_observed_results=[
                "File inspected [E0001,E0002]; tool content returned [E0003].",
                "User added instruction [E0004].",
            ],
            important_identifiers=[],
            contradictions_or_missing_information=["Count is an assistant claim [E0005]."],
            source_event_ids=["E0001", "E0002", "E0003", "E0004", "E0005"],
        ),
        separators=(",", ":"),
    )
    representations = dict(
        full=body,
        head_tail=head_tail(body, budget, count),
        free_summary=free,
        structured_summary=structured,
    )
    if budget == count(body):
        representations = {key: body for key in representations}
    else:
        for condition in ("free_summary", "structured_summary"):
            validate_summary(
                representations[condition],
                condition,
                budget,
                count,
                {event.event_id for event in transcript.events},
            )
    artifact = dict(
        data_origin="synthetic_fixture",
        banner="SYNTHETIC FIXTURE — NOT EMPIRICAL DATA",
        summary_origin="independently hand-authored teaching examples, not API outputs",
        counter_method="synthetic-only ceil(utf8_bytes/4); not a model token count",
        initial_task_header=render_task_header(transcript),
        budget=budget,
        body_tokens=count(body),
        compression_applied=budget < count(body),
        representations=representations,
        measured_tokens={k: count(v) for k, v in representations.items()},
        generation_calls=0,
        empirical_scores_generated=False,
    )
    PrivateStore(Path("runs/private/demo")).put("artifacts", "representations", artifact)
    return {
        key: value
        for key, value in artifact.items()
        if key not in ("representations", "initial_task_header")
    }


def _dataset_manifest(config):
    private = Path(config.dataset_dir) / "manifest.json"
    if not private.exists():
        raise ValueError("Real data missing; run acquire and inventory explicitly first")
    data = json.loads(private.read_text())
    # Only allowlisted non-text aggregate provenance enters protocol/public manifests.
    keys = (
        "schema_version",
        "data_origin",
        "dataset_commit",
        "source_repository",
        "split_seed",
        "counts",
        "exclusions",
        "family_rule",
        "model_input_view",
        "initial_task_rule",
        "body_size",
        "normalized_content_digest",
        "labels_sha256",
    )
    return {k: data[k] for k in keys if k in data}


def freeze(config, development_run: Path) -> dict:
    import hashlib

    import pandas as pd

    from context_audit.dataset import load_dataset
    from context_audit.metrics import validate_rows
    from context_audit.provider import utc_now
    from context_audit.representations import CONDITIONS
    from context_audit.runner import protocol_signature, require_private_path
    from context_audit.storage import digest

    if config.split != "test" or config.protocol_version != "protocol-v1":
        raise ValueError("Freeze requires the complete test protocol-v1 configuration")
    if not config.rubric_reviewed or not config.data_use_confirmed:
        raise ValueError("Review rubric/hypothesis and confirm data use before freezing")
    require_private_path(development_run, Path("runs/private"))
    completion_file = development_run / "manifests/completion.json"
    manifest_file = development_run / "manifests/run.json"
    if not completion_file.exists() or not manifest_file.exists():
        raise ValueError("Freeze requires a completed, reviewed real development run")
    completion = json.loads(completion_file.read_text())
    development = json.loads(manifest_file.read_text())
    _, dev_labels = load_dataset(Path(config.dataset_dir), "development")
    planned = {x["transcript_id"] for x in development["planned_calls"]}
    if (
        completion["status"] != "executed"
        or development["data_origin"] != "sleight_bench"
        or development["config"]["split"] != "development"
        or planned != {x.transcript_id for x in dev_labels}
    ):
        raise ValueError("Complete all development pairs successfully before freezing test methods")
    expected_schedule = {
        (label.transcript_id, condition, repetition)
        for label in dev_labels
        for condition in CONDITIONS
        for repetition in range(config.repetitions)
    }
    actual_schedule = [
        (item["transcript_id"], item["condition"], item["repetition"])
        for item in development["planned_calls"]
    ]
    if len(actual_schedule) != len(expected_schedule) or set(actual_schedule) != expected_schedule:
        raise ValueError("Development must cover every pair, condition and planned repetition")
    score_path = development_run / "scores.csv"
    if (
        not score_path.exists()
        or completion.get("scores_sha256") != hashlib.sha256(score_path.read_bytes()).hexdigest()
    ):
        raise ValueError("Development numeric scores are missing or changed after completion")
    scores = validate_rows(pd.read_csv(score_path))
    units = set(zip(scores.transcript_id, scores.condition, scores.repetition, strict=True))
    if (
        units != expected_schedule
        or not scores.status.eq("ok").all()
        or not scores.data_origin.eq("sleight_bench").all()
        or not scores.split.eq("development").all()
    ):
        raise ValueError(
            "Development numeric scores do not prove complete successful real evaluation"
        )
    dataset_manifest = _dataset_manifest(config)
    signature = protocol_signature(config, dataset_manifest)
    if development.get("code_hash") != signature["code_hash"]:
        raise ValueError("Code changed since completed development; rerun development validation")
    if development.get("dataset_manifest_hash") != digest(dataset_manifest):
        raise ValueError(
            "Dataset changed since completed development; rerun development validation"
        )
    expected_rows = len(development["planned_calls"])
    if (
        completion.get("run_id") != development["run_id"]
        or completion.get("rows") != expected_rows
        or completion.get("expected_rows") != expected_rows
        or completion.get("successful_rows") != expected_rows
    ):
        raise ValueError("Development completion does not match its planned successful run")
    if development["prompt_hashes"] != signature["prompts"]:
        raise ValueError("Prompts differ from the completed development run")
    compare = (
        "provider",
        "qwen",
        "monitor_model",
        "summarizer_model",
        "monitor_context_window",
        "summarizer_context_window",
        "monitor_max_tokens",
        "summary_max_tokens",
        "token_fraction",
        "token_minimum",
        "token_maximum",
        "threshold",
        "max_attempts",
        "repetitions",
        "concurrency",
        "bootstrap_seed",
        "bootstrap_samples",
    )
    if any(development["config"].get(k) != config.model_dump()[k] for k in compare):
        raise ValueError("Test methods differ from the completed development run")
    target = Path(config.protocol_file)
    artifact = dict(
        protocol_version="protocol-v1",
        frozen_at=utc_now(),
        signature=signature,
        development_run_id=development["run_id"],
        review="User explicitly supplied --reviewed after reviewing rubric and pilot",
    )
    if target.exists():
        if json.loads(target.read_text())["signature"] != signature:
            raise ValueError("Existing protocol-v1 differs; never silently overwrite a freeze")
        return dict(status="already_frozen", protocol_file=str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2) + "\n")
    return dict(
        status="freeze_prepared",
        protocol_file=str(target),
        next_step="Review and commit protocol/code/config/prompts, then tag HEAD protocol-v1. "
        "Test runner requires that tag and matching content hashes.",
    )


def export_results(run_dir: Path, output: Path) -> dict:
    import pandas as pd

    from context_audit.dataset import load_dataset
    from context_audit.metrics import validate_rows
    from context_audit.reporting import _validate_public_identifiers
    from context_audit.runner import require_private_path
    from context_audit.runtime_models import AuditConfig
    from context_audit.storage import digest

    require_private_path(run_dir, Path("runs/private"))
    manifest = json.loads((run_dir / "manifests/run.json").read_text())
    if manifest["data_origin"] != "sleight_bench":
        raise ValueError("Cannot export synthetic fixtures as empirical data")
    current_manifest = _dataset_manifest(AuditConfig.model_validate(manifest["config"]))
    if digest(current_manifest) != manifest["dataset_manifest_hash"]:
        raise ValueError("Dataset changed since scoring; cannot export different evaluator labels")
    frame = validate_rows(pd.read_csv(run_dir / "scores.csv"), allow_partial_pairs=True)
    _validate_public_identifiers(frame)
    if not frame.empty and not frame.data_origin.eq("sleight_bench").all():
        raise ValueError("Cannot export synthetic scores as empirical data")
    _, labels = load_dataset(Path(manifest["config"]["dataset_dir"]), manifest["config"]["split"])
    selected = set(manifest["planned_transcripts"])
    public = {
        k: manifest[k]
        for k in (
            "run_id",
            "dataset_commit",
            "code_commit",
            "code_hash",
            "seeds",
            "prompt_hashes",
            "models",
            "failure_policy",
            "created_at",
            "prices",
            "data_origin",
            "planned_transcripts",
            "planned_calls",
            "dataset_manifest_hash",
            "counter_method",
            "concurrency",
        )
    }
    public["evaluation_labels"] = [
        label.model_dump() for label in labels if label.transcript_id in selected
    ]
    public["config"] = {
        k: v
        for k, v in manifest["config"].items()
        if k not in ("dataset_dir", "run_dir", "prompt_dir", "prices_file", "protocol_file")
    }
    gpu_receipt = run_dir / "manifests/gpu_costs.json"
    if gpu_receipt.exists():
        public["gpu_costs"] = json.loads(gpu_receipt.read_text())
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "public_scores.csv", index=False)
    (output / "run_manifest.json").write_text(json.dumps(public, indent=2) + "\n")
    return dict(status="exported_local_numeric_data", rows=len(frame), output=str(output))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="Independent fixtures, zero API calls, no empirical scores")
    acquire = sub.add_parser(
        "acquire", help="Pinned official clone and documented local decryption"
    )
    acquire.add_argument("--private-dir", type=Path, default=Path("data/private"))
    inventory = sub.add_parser("inventory", help="Build opaque inventory and whole-family split")
    inventory.add_argument("--upstream", type=Path, default=Path("data/private/upstream"))
    inventory.add_argument("--private-dir", type=Path, default=Path("data/private"))
    inventory.add_argument("--public-dir", type=Path, default=Path("data/manifests"))
    inventory.add_argument("--seed", type=int, default=20260905)
    inventory.add_argument("--development-pairs", type=int, default=8)
    run = sub.add_parser("run", help="Real live evaluation (never automatic on notebook open)")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--live", action="store_true", help="Explicit paid-generation opt-in")
    run.add_argument("--max-cost-usd", type=float, required=True)
    frozen = sub.add_parser("freeze", help="Prepare protocol freeze after reviewed development")
    frozen.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    frozen.add_argument("--development-run", type=Path, required=True)
    frozen.add_argument("--reviewed", action="store_true")
    export = sub.add_parser("export", help="Create only local sanitized empirical result artifacts")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--output", type=Path, default=Path("results"))
    analyze = sub.add_parser("analyze", help="Offline real numeric results only, zero API calls")
    analyze.add_argument("--scores", type=Path, default=Path("results/public_scores.csv"))
    analyze.add_argument("--output", type=Path, default=Path("results"))
    analyze.add_argument("--manifest", type=Path, default=Path("results/run_manifest.json"))
    analyze.add_argument("--bootstrap-samples", type=int, default=None)
    analyze.add_argument("--seed", type=int, default=None)
    sub.add_parser("scan-public", help="Scan Git candidate files for protected content and secrets")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            result = demo()
        elif args.command == "acquire":
            from context_audit.dataset import acquire_dataset

            result = acquire_dataset(args.private_dir)
        elif args.command == "inventory":
            from context_audit.dataset import inventory_dataset

            result = inventory_dataset(
                args.upstream,
                args.private_dir,
                args.public_dir,
                seed=args.seed,
                development_pairs=args.development_pairs,
            )
        elif args.command == "run":
            from context_audit.dataset import load_canaries, load_dataset
            from context_audit.runner import load_config, run_experiment

            if not args.live or not math.isfinite(args.max_cost_usd) or args.max_cost_usd <= 0:
                raise ValueError("Live run requires --live and a positive finite --max-cost-usd")
            load_dotenv(override=False)
            config = load_config(args.config)
            if config.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY is missing; no paid call was made")
            inputs, labels = load_dataset(Path(config.dataset_dir), config.split)
            result = run_experiment(
                config,
                inputs,
                labels,
                _dataset_manifest(config),
                max_cost_usd=args.max_cost_usd,
                canaries=load_canaries(Path(config.dataset_dir)),
            )
        elif args.command == "freeze":
            from context_audit.runner import load_config

            if not args.reviewed:
                raise ValueError(
                    "Review hypothesis/rubric/pilot first, then explicitly pass --reviewed"
                )
            result = freeze(load_config(args.config), args.development_run)
        elif args.command == "export":
            result = export_results(args.run_dir, args.output)
        elif args.command == "analyze":
            from context_audit.reporting import generate_report

            result = generate_report(
                args.scores,
                args.output,
                n_bootstrap=args.bootstrap_samples,
                seed=args.seed,
                manifest_path=args.manifest if args.manifest.exists() else None,
            )
        else:
            from context_audit.sanitation import scan_repository

            result = scan_repository()
        print(json.dumps(result, indent=2, allow_nan=False))
        return 1 if result.get("status") in ("failed", "partial_or_failed") else 0
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        print(f"context-audit: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
