"""Versioned Colab reruns preserve independent synthetic prior-study artifacts."""

import json
import runpy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/colab_bootstrap.py"


def bootstrap(root, **settings):
    namespace = runpy.run_path(str(SCRIPT))["run_guided"].__globals__
    namespace.update(
        DRIVE_ROOT=root, REPO=root / "local-source", SETUP_READY=True,
        DATA_USE_CONFIRMED=True, RUBRIC_REVIEWED=True, **settings,
    )
    return namespace


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def previous_study(tmp_path):
    """No benchmark text, GPU, network, or generated model output is involved."""
    write_json(tmp_path / "configuration/code-pin.json", {
        "repo_url": "https://example.invalid/synthetic-study.git",
        "branch": "pilot", "commit": "a" * 40,
    })
    write_json(tmp_path / "configuration/model-pin.json", {
        "model_id": "Qwen/Qwen3.8-27B", "model_revision": "b" * 40,
        "vllm_version": "0.28.0", "runtime_versions": {},
    })
    write_json(tmp_path / "configuration/context/selection.json", {
        "context_window": 196608, "previous_context_window": 65536,
        "source_run": "runs-private/qwen-pilot", "source_run_id": "synthetic-prior",
    })
    write_json(tmp_path / "public-manifests/inventory.json", {"synthetic_pairs": 3})
    write_json(tmp_path / "data-private/manifest.json", {"synthetic_data": True})
    write_json(tmp_path / "runs-private/qwen-pilot-ctx196608/manifests/run.json", {
        "run_id": "synthetic-prior", "config": {"split": "development"},
    })
    write_json(tmp_path / "configuration/pilot-ctx196608.json", {"legacy_config": True})
    old_report = tmp_path / "numeric-results/pilot/reproduced/findings.md"
    old_report.parent.mkdir(parents=True)
    old_report.write_text("Independent prior-study report must remain unchanged.\n")
    from context_audit.colab import initialize_gpu_budget

    initialize_gpu_budget(tmp_path / "runs-private/qwen-pilot", 12,
                          previously_used_seconds=1800)
    return tmp_path


def snapshot(directory):
    return {p.relative_to(directory): p.read_bytes()
            for p in directory.rglob("*") if p.is_file()}


def test_default_version_preserves_legacy_routing_without_creating_a_version(previous_study):
    ns = bootstrap(previous_study)
    before = snapshot(previous_study)
    assert ns["EXPERIMENT_VERSION"] == "legacy"
    assert ns["source_workspace"]() == previous_study
    ns["prepare_version_workspace"]()
    assert ns["phase_settings"]("pilot") == ("pilot-ctx196608", 196608)
    assert snapshot(previous_study) == before


@pytest.mark.parametrize("invalid", ["", "summary-v3", "../outside", "/tmp/outside"])
def test_arbitrary_versions_fail_before_writing_files(previous_study, invalid):
    ns = bootstrap(previous_study, EXPERIMENT_VERSION=invalid)
    before = snapshot(previous_study)
    with pytest.raises(ValueError):
        ns["prepare_version_workspace"]()
    assert snapshot(previous_study) == before


def test_new_version_inherits_pins_and_context_without_modifying_the_parent(previous_study):
    root = previous_study
    ns = bootstrap(root, EXPERIMENT_VERSION="summary-v2")
    before = snapshot(root)
    before_budget = ns["read_budget"](root)
    ns["prepare_version_workspace"]()
    version = root / "versions/summary-v2"
    assert ns["source_workspace"]() == version
    assert (version / "configuration/version.json").is_file()
    for relative in ("configuration/code-pin.json", "configuration/context/selection.json",
                     "public-manifests/inventory.json"):
        assert (version / relative).read_bytes() == (root / relative).read_bytes()
        assert not (version / relative).is_symlink()
    for path, contents in before.items():
        assert (root / path).read_bytes() == contents
    assert not (version / "configuration/model-pin.json").exists()
    assert not (version / "data-private").exists()
    assert not (version / "runs-private").exists()
    assert not list(version.rglob("gpu_budget"))
    assert not any(path.is_symlink() for path in version.rglob("*"))
    assert ns["read_budget"](root) == before_budget


def test_reconnect_does_not_reimport_changed_parent_state(previous_study):
    root = previous_study
    ns = bootstrap(root, EXPERIMENT_VERSION="summary-v2")
    ns["prepare_version_workspace"]()
    version = ns["source_workspace"]()
    before = snapshot(version)
    write_json(root / "configuration/context/selection.json", {"context_window": 262144})
    write_json(root / "configuration/code-pin.json", {"commit": "c" * 40})
    write_json(root / "public-manifests/inventory.json", {"synthetic_pairs": 99})
    resumed = bootstrap(root, EXPERIMENT_VERSION="summary-v2")
    resumed["prepare_version_workspace"]()
    assert snapshot(version) == before
    assert resumed["phase_settings"]("pilot") == ("pilot-summary-v2-ctx196608", 196608)


@pytest.mark.parametrize("marker", [
    "frozen-source.zip", "public-manifests/protocol-v1.json",
    "runs-private/qwen-test/manifests/run.json",
    "runs-private/qwen-test-ctx196608/manifests/run.json",
])
def test_frozen_or_test_parent_cannot_be_used_to_start_new_version(previous_study, marker):
    root = previous_study
    write_json(root / marker, {"config": {"split": "test"}})
    before = snapshot(root)
    ns = bootstrap(root, EXPERIMENT_VERSION="summary-v2")
    with pytest.raises(ValueError, match="frozen|Frozen|test|review"):
        ns["prepare_version_workspace"]()
    assert snapshot(root) == before
    assert not (root / "versions/summary-v2/configuration/version.json").exists()


def test_missing_legacy_workspace_creates_isolated_defaults(tmp_path):
    ns = bootstrap(tmp_path, EXPERIMENT_VERSION="summary-v2")
    ns["prepare_version_workspace"]()
    assert ns["source_workspace"]() == tmp_path / "versions/summary-v2"
    assert ns["phase_settings"]("pilot") == ("pilot-summary-v2", 65536)
    assert not (tmp_path / "configuration/context/selection.json").exists()


@pytest.mark.parametrize("phase", ["pilot", "development", "test"])
def test_configuration_routes_each_phase_and_preserves_model_data_and_budget(
    previous_study, phase,
):
    root = previous_study
    ns = bootstrap(root, EXPERIMENT_VERSION="summary-v2")
    ns["prepare_version_workspace"]()
    old_files = snapshot(root)
    name = f"{phase}-summary-v2-ctx196608"
    assert ns["phase_settings"](phase) == (name, 196608)
    assert ns["phase_run_dir"](phase) == Path(f"runs/private/qwen-{name}")
    config = ns["configured_phase"](phase)
    expected_protocol = "protocol-v1" if phase == "test" else "development-summary-v2"
    assert config.protocol_version == expected_protocol
    assert config.run_dir == f"runs/private/qwen-{name}"
    assert config.split == ("test" if phase == "test" else "development")
    assert config.dataset_dir == "data/private"
    assert config.qwen.model_revision == "b" * 40
    assert config.qwen.gpu_budget_hours == 12
    assert config.qwen.max_model_len == 196608
    assert config.monitor_context_window == config.summarizer_context_window == 196608
    assert config.pilot_pairs == (3 if phase == "pilot" else None)
    saved = root / "versions/summary-v2/configuration" / f"{name}.json"
    assert json.loads(saved.read_text()) == config.model_dump()
    assert not (root / "configuration" / f"{name}.json").exists()
    for relative, contents in old_files.items():
        assert (root / relative).read_bytes() == contents
    assert ns["configured_phase"](phase).model_dump() == config.model_dump()


def test_version_reports_and_diagnostics_use_new_run_and_leave_old_results(
    previous_study, monkeypatch,
):
    import subprocess

    root = previous_study
    ns = bootstrap(root, EXPERIMENT_VERSION="summary-v2")
    ns["prepare_version_workspace"]()
    monkeypatch.chdir(root)
    (root / "runs").mkdir()
    # Colab's local checkout can bind private directories; the Drive version itself cannot.
    (root / "runs/private").symlink_to(root / "runs-private", target_is_directory=True)
    current = root / "runs-private/qwen-pilot-summary-v2-ctx196608"
    logs = current / "gpu_sessions/runner_logs"
    logs.mkdir(parents=True)
    (logs / "synthetic.json").write_text("Independent new-version diagnostic")
    (current / "scores.csv").write_text("Synthetic rows: export substituted at child boundary")
    old_report = root / "numeric-results/pilot/reproduced/findings.md"
    old_contents = old_report.read_bytes()
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if "analyze" in command:
            output = Path(command[command.index("--output") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "metrics.json").write_text('{"status": "insufficient_data"}')

    monkeypatch.setattr(subprocess, "run", run)
    assert "new-version diagnostic" in ns["private_log_tail"]("pilot")
    result = ns["export_phase"]("pilot", 120)
    export = commands[0]
    assert Path(export[export.index("--run-dir") + 1]).resolve() == current.resolve()
    assert result["scores"] == str(root / "numeric-results/summary-v2/pilot/public_scores.csv")
    assert result["report"] == str(root / "numeric-results/summary-v2/pilot/reproduced/findings.md")
    assert old_report.read_bytes() == old_contents


def test_source_setup_separates_git_and_receipts_but_binds_shared_private_data(previous_study):
    root = previous_study
    ns = bootstrap(root, EXPERIMENT_VERSION="summary-v2")
    version = root / "versions/summary-v2"
    source = ns["REPO"] / "src/context_audit/colab.py"
    source.parent.mkdir(parents=True)
    source.write_text('"""Independent source placeholder; never executed."""\n')
    calls = {}
    bindings = []
    ns.update(
        checkout_source=lambda *args: calls.update(checkout=args),
        bind_git_metadata=lambda *args: calls.update(git=args),
        bind_private_directory=lambda *args: bindings.append(args),
        apply_embedded_source=lambda *args, **kwargs: calls.update(
            snapshot_args=args, snapshot_kwargs=kwargs,
        ),
    )
    ns["prepare_source"]()
    assert calls["checkout"][-1] == version / "configuration/code-pin.json"
    assert calls["git"] == (ns["REPO"], version / "git-metadata", "a" * 40)
    assert bindings == [
        ("data/private", root / "data-private"),
        ("runs/private", root / "runs-private"),
    ]
    assert calls["snapshot_args"] == (ns["REPO"], version)
    assert calls["snapshot_kwargs"] == {
        "frozen": False, "run_root": root / "runs-private", "run_version": "summary-v2",
    }
    assert (ns["REPO"] / "data/manifests/inventory.json").read_bytes() == (
        version / "public-manifests/inventory.json"
    ).read_bytes()


def test_saving_public_manifests_cannot_overwrite_parent_inventory(previous_study):
    root = previous_study
    ns = bootstrap(root, EXPERIMENT_VERSION="summary-v2")
    ns["prepare_version_workspace"]()
    original = (root / "public-manifests/inventory.json").read_bytes()
    source = ns["REPO"] / "data/manifests/inventory.json"
    write_json(source, {"synthetic_new_version": True})
    ns["save_public_manifests"]()
    assert (root / "public-manifests/inventory.json").read_bytes() == original
    assert (root / "versions/summary-v2/public-manifests/inventory.json").read_bytes() == (
        source.read_bytes()
    )
