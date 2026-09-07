"""Context recovery uses independent numeric fixtures, never benchmark content."""

import json
import runpy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/colab_bootstrap.py"


@pytest.fixture
def preflight(tmp_path, monkeypatch):
    from context_audit import cli, runner

    ns = runpy.run_path(str(SCRIPT))["run_guided"].__globals__
    ns.update(DRIVE_ROOT=tmp_path, REPO=tmp_path, SETUP_READY=True,
              DATA_USE_CONFIRMED=True, RUBRIC_REVIEWED=True)
    configuration = tmp_path / "configuration"
    configuration.mkdir()
    (configuration / "model-pin.json").write_text(json.dumps({
        "model_id": "Qwen/Qwen3.8-27B", "model_revision": "a" * 40,
        "vllm_version": "0.28.0", "runtime_versions": {},
    }))
    config = ns["configured_phase"]("pilot")
    directory = tmp_path / "runs-private/qwen-pilot"
    manifests = directory / "manifests"
    manifests.mkdir(parents=True)
    signature = {"code_hash": "synthetic-code", "prompts": {"monitor": "synthetic"},
                 "dataset_manifest_hash": "synthetic-data"}
    monkeypatch.setattr(cli, "_dataset_manifest", lambda config: {
        "counts": {"eligible_transcripts": 2},
    })
    monkeypatch.setattr(runner, "protocol_signature", lambda *args: signature.copy())
    (manifests / "run.json").write_text(json.dumps({
        "run_id": "synthetic-run", "config": config.model_dump(),
        "code_hash": signature["code_hash"], "prompt_hashes": signature["prompts"],
        "dataset_manifest_hash": signature["dataset_manifest_hash"],
    }))
    inventory = {"items": [
        {"transcript_id": "synthetic-short", "body_tokens": 100,
         "full_input_tokens": 200, "summary_input_tokens": 300,
         "monitor_window": 65536, "summary_window": 65536},
        {"transcript_id": "synthetic-long", "body_tokens": 60000,
         "full_input_tokens": 64000, "summary_input_tokens": 130000,
         "monitor_window": 65536, "summary_window": 65536},
    ], "context_limit_ids": ["synthetic-long"]}
    (manifests / "preflight.json").write_text(json.dumps(inventory))
    ledger = tmp_path / "runs-private/gpu_budget/budget-seconds.jsonl"
    ledger.parent.mkdir()
    ledger.write_text("Independent fixture for unchanged allocation receipts\n")
    return ns, directory, inventory


def prepare(ns):
    assert "prepare_context_window" in ns, "Saved context failures need automatic recovery"
    return ns["prepare_context_window"]()


def test_context_recovery_reserves_output_and_preserves_previous_attempt(preflight, capsys):
    ns, directory, _ = preflight
    root = ns["DRIVE_ROOT"]
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert prepare(ns) is True
    for path, contents in before.items():
        assert path.read_bytes() == contents
    # 130,000 prompt tokens + 1,600 output exceed 131,072; next 32K block is 163,840.
    for phase in ("pilot", "development", "test"):
        config = ns["configured_phase"](phase)
        assert config.qwen.max_model_len == 163840
        assert config.monitor_context_window == config.summarizer_context_window == 163840
        assert config.run_dir == f"runs/private/qwen-{phase}-ctx163840"
    assert directory.exists()
    assert prepare(ns) is False
    output = capsys.readouterr().out
    assert "131600" in output and "163840" in output
    assert "synthetic-long" not in output
    assert not list((root / "runs-private").glob("qwen-*/calls/*"))


def test_reconnect_uses_saved_selection_without_rewriting_it(preflight):
    ns, _, _ = preflight
    prepare(ns)
    selection = ns["DRIVE_ROOT"] / "configuration/context/selection.json"
    before = selection.read_bytes()
    fresh = runpy.run_path(str(SCRIPT))["run_guided"].__globals__
    fresh.update(DRIVE_ROOT=ns["DRIVE_ROOT"], REPO=ns["REPO"], SETUP_READY=True,
                 DATA_USE_CONFIRMED=True, RUBRIC_REVIEWED=True)
    assert fresh["configured_phase"]("pilot").qwen.max_model_len == 163840
    assert fresh["prepare_context_window"]() is False
    assert selection.read_bytes() == before


def test_guided_resume_selects_context_before_launching_the_model(preflight):
    from types import SimpleNamespace

    ns, _, _ = preflight
    launched = []
    ns.update(
        START_RUN=True, mount_workspace=lambda: None, check_gpu=lambda: None,
        prepare_source=lambda: None, acquire_data=lambda: None, release_gpu=lambda: None,
        install_runtime=lambda: ns.update(SETUP_READY=True),
        NotebookAllocation=lambda *args: SimpleNamespace(
            checkpoint=lambda: {"remaining_seconds": 1000}, remaining=lambda: 1000,
            close=lambda: None,
        ),
        execute_phase=lambda config, seconds: (
            launched.append((config, seconds)) or {"status": "executed"}
        ),
    )
    result = ns["run_guided"]()
    assert len(launched) == 1
    config, seconds = launched[0]
    assert config.qwen.max_model_len == 163840
    assert config.run_dir == "runs/private/qwen-pilot-ctx163840"
    assert seconds == 850
    assert result["phases"]["pilot"]["outputs"]["run_dir"] == config.run_dir


def test_reports_and_diagnostics_read_the_selected_attempt(preflight, monkeypatch):
    import subprocess

    ns, _, _ = preflight
    prepare(ns)
    root = ns["DRIVE_ROOT"]
    monkeypatch.chdir(root)
    (root / "runs").mkdir()
    (root / "runs/private").symlink_to(root / "runs-private", target_is_directory=True)
    active = root / "runs-private/qwen-pilot-ctx163840"
    logs = active / "gpu_sessions/runner_logs"
    logs.mkdir(parents=True)
    (logs / "fixture.json").write_text("Synthetic current attempt diagnostic")
    assert "current attempt" in ns["private_log_tail"]("pilot")
    (active / "scores.csv").write_text("Synthetic rows: CLI substituted at subprocess boundary")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if "analyze" in command:
            report = root / "numeric-results/pilot/reproduced"
            report.mkdir(parents=True)
            (report / "metrics.json").write_text('{"status": "insufficient_data"}')

    monkeypatch.setattr(subprocess, "run", run)
    result = ns["export_phase"]("pilot", 120)
    assert Path(commands[0][commands[0].index("--run-dir") + 1]).resolve() == active.resolve()
    assert result["scores"].endswith("numeric-results/pilot/public_scores.csv")


def test_freeze_reads_development_from_the_same_context_selection(preflight, monkeypatch):
    from context_audit import cli

    ns, _, _ = preflight
    prepare(ns)
    received = []

    def freeze(config, development):
        received.append((config, development))
        raise ValueError("Independent stop before Git freeze operations")

    monkeypatch.setattr(cli, "freeze", freeze)
    with pytest.raises(ValueError, match="Independent stop"):
        ns["freeze_reviewed"]()
    config, development = received[0]
    assert config.qwen.max_model_len == 163840
    assert development == Path("runs/private/qwen-development-ctx163840")


@pytest.mark.parametrize("artifact", [
    "calls/attempt.json", "requests/attempt.json", "representations/item.json",
    "results/item.json", "scores.csv", "manifests/completion.json", "budget-seconds.jsonl",
    "../qwen-development/calls/attempt.json", "../qwen-test/requests/attempt.json",
])
def test_generation_evidence_blocks_context_recovery_without_erasing_it(preflight, artifact):
    ns, directory, _ = preflight
    evidence = directory / artifact
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text("Independent evidence must be retained")
    with pytest.raises(ValueError, match="generation|review|frozen"):
        prepare(ns)
    assert evidence.read_text() == "Independent evidence must be retained"
    assert not (ns["DRIVE_ROOT"] / "configuration/context/selection.json").exists()


@pytest.mark.parametrize("change", ["too_long", "incomplete", "duplicates", "wrong_ids",
                                    "negative", "wrong_window", "wrong_source"])
def test_invalid_or_unsupported_inventory_never_selects_a_smaller_context(preflight, change):
    ns, directory, inventory = preflight
    if change == "too_long":
        inventory["items"][1]["summary_input_tokens"] = 262000
    elif change == "incomplete":
        inventory["items"].pop(0)
    elif change == "duplicates":
        inventory["items"][0]["transcript_id"] = "synthetic-long"
    elif change == "wrong_ids":
        inventory["context_limit_ids"] = ["not-in-inventory"]
    elif change == "negative":
        inventory["items"][0]["full_input_tokens"] = -1
    elif change == "wrong_window":
        inventory["items"][0]["monitor_window"] = 32768
    else:
        manifest = directory / "manifests/run.json"
        data = json.loads(manifest.read_text())
        data["code_hash"] = "different-source"
        manifest.write_text(json.dumps(data))
    (directory / "manifests/preflight.json").write_text(json.dumps(inventory))
    with pytest.raises(ValueError):
        prepare(ns)
    assert not (ns["DRIVE_ROOT"] / "configuration/context/selection.json").exists()


@pytest.mark.parametrize("frozen", ["frozen-source.zip", "public-manifests/protocol-v1.json"])
def test_frozen_protocol_prevents_automatic_context_changes(preflight, frozen):
    ns, _, _ = preflight
    path = ns["DRIVE_ROOT"] / frozen
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Independent frozen marker")
    with pytest.raises(ValueError, match="frozen|Frozen"):
        prepare(ns)


def test_test_stage_cannot_trigger_context_selection(preflight):
    ns, _, _ = preflight
    ns["STAGE"] = "test"
    assert prepare(ns) is False
    assert not (ns["DRIVE_ROOT"] / "configuration/context/selection.json").exists()


def test_new_workspace_without_preflight_keeps_existing_configuration(preflight):
    ns, directory, _ = preflight
    (directory / "manifests/preflight.json").unlink()
    assert prepare(ns) is False
    assert ns["configured_phase"]("pilot").qwen.max_model_len == 65536


@pytest.fixture
def v3_preflight(preflight):
    ns, previous, inventory = preflight
    ns["EXPERIMENT_VERSION"] = "summary-v3"
    ns["prepare_version_workspace"]()
    config = ns["configured_phase"]("pilot")
    directory = ns["DRIVE_ROOT"] / "runs-private/qwen-pilot-summary-v3"
    manifests = directory / "manifests"
    manifests.mkdir(parents=True)
    saved = json.loads((previous / "manifests/run.json").read_text())
    saved.update(config=config.model_dump(), run_id="synthetic-v3-context-run")
    (manifests / "run.json").write_text(json.dumps(saved))
    (manifests / "preflight.json").write_text(json.dumps(inventory))
    # Earlier version requests remain private evidence, not new-version generation.
    for name in ("qwen-pilot", "qwen-pilot-summary-v2-ctx196608"):
        calls = ns["DRIVE_ROOT"] / "runs-private" / name / "calls"
        calls.mkdir(parents=True, exist_ok=True)
        (calls / "synthetic.json").write_text("Earlier generation must remain unchanged")
    return ns, directory


def test_v3_context_recovery_preserves_old_generation_and_parent_context(v3_preflight):
    ns, directory = v3_preflight
    root = ns["DRIVE_ROOT"]
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert prepare(ns) is True
    for path, contents in before.items():
        assert path.read_bytes() == contents
    assert not (root / "configuration/context/selection.json").exists()
    assert not (root / "versions/summary-v2/configuration/context/selection.json").exists()
    assert (root / "versions/summary-v3/configuration/context/selection.json").is_file()
    for phase in ("pilot", "development", "test"):
        config = ns["configured_phase"](phase)
        assert config.qwen.max_model_len == 163840
        assert config.run_dir == f"runs/private/qwen-{phase}-summary-v3-ctx163840"
        assert config.structured_summary_mode == "schema_citations_v1"
    assert directory.is_dir()


def test_v3_context_recovery_still_rejects_its_own_generation(v3_preflight):
    ns, directory = v3_preflight
    calls = directory / "calls"
    calls.mkdir()
    (calls / "synthetic.json").write_text("New-version generation also requires review")
    with pytest.raises(ValueError, match="generation|review"):
        prepare(ns)
    assert not (ns["source_workspace"]() / "configuration/context/selection.json").exists()
