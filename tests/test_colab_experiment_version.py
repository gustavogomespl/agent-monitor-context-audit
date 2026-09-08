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


@pytest.fixture(params=["summary-v2", "summary-v3", "summary-v4", "summary-v5", "summary-v6"])
def experiment_version(request):
    return request.param


def test_default_version_preserves_legacy_routing_without_creating_a_version(previous_study):
    ns = bootstrap(previous_study)
    before = snapshot(previous_study)
    assert ns["EXPERIMENT_VERSION"] == "legacy"
    assert ns["source_workspace"]() == previous_study
    ns["prepare_version_workspace"]()
    assert ns["phase_settings"]("pilot") == ("pilot-ctx196608", 196608)
    assert snapshot(previous_study) == before


@pytest.mark.parametrize("invalid", ["", "summary-v7", "../outside", "/tmp/outside"])
def test_arbitrary_versions_fail_before_writing_files(previous_study, invalid):
    ns = bootstrap(previous_study, EXPERIMENT_VERSION=invalid)
    before = snapshot(previous_study)
    with pytest.raises(ValueError):
        ns["prepare_version_workspace"]()
    assert snapshot(previous_study) == before


def test_new_version_inherits_pins_and_context_without_modifying_the_parent(
    previous_study, experiment_version,
):
    root = previous_study
    ns = bootstrap(root, EXPERIMENT_VERSION=experiment_version)
    before = snapshot(root)
    before_budget = ns["read_budget"](root)
    ns["prepare_version_workspace"]()
    version = root / "versions" / experiment_version
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


def test_reconnect_does_not_reimport_changed_parent_state(previous_study, experiment_version):
    root = previous_study
    ns = bootstrap(root, EXPERIMENT_VERSION=experiment_version)
    ns["prepare_version_workspace"]()
    version = ns["source_workspace"]()
    before = snapshot(version)
    write_json(root / "configuration/context/selection.json", {"context_window": 262144})
    write_json(root / "configuration/code-pin.json", {"commit": "c" * 40})
    write_json(root / "public-manifests/inventory.json", {"synthetic_pairs": 99})
    resumed = bootstrap(root, EXPERIMENT_VERSION=experiment_version)
    resumed["prepare_version_workspace"]()
    assert snapshot(version) == before
    assert resumed["phase_settings"]("pilot") == (f"pilot-{experiment_version}-ctx196608", 196608)


@pytest.mark.parametrize("marker", [
    "frozen-source.zip", "public-manifests/protocol-v1.json",
    "runs-private/qwen-test/manifests/run.json",
    "runs-private/qwen-test-ctx196608/manifests/run.json",
])
def test_frozen_or_test_parent_cannot_be_used_to_start_new_version(
    previous_study, marker, experiment_version,
):
    root = previous_study
    write_json(root / marker, {"config": {"split": "test"}})
    before = snapshot(root)
    ns = bootstrap(root, EXPERIMENT_VERSION=experiment_version)
    with pytest.raises(ValueError, match="frozen|Frozen|test|review"):
        ns["prepare_version_workspace"]()
    assert snapshot(root) == before
    assert not (root / "versions" / experiment_version / "configuration/version.json").exists()


def test_missing_legacy_workspace_creates_isolated_defaults(tmp_path, experiment_version):
    ns = bootstrap(tmp_path, EXPERIMENT_VERSION=experiment_version)
    ns["prepare_version_workspace"]()
    assert ns["source_workspace"]() == tmp_path / "versions" / experiment_version
    assert ns["phase_settings"]("pilot") == (f"pilot-{experiment_version}", 65536)
    assert not (tmp_path / "configuration/context/selection.json").exists()


@pytest.mark.parametrize("phase", ["pilot", "development", "test"])
def test_configuration_routes_each_phase_and_preserves_model_data_and_budget(
    previous_study, phase, experiment_version,
):
    root = previous_study
    ns = bootstrap(root, EXPERIMENT_VERSION=experiment_version)
    ns["prepare_version_workspace"]()
    old_files = snapshot(root)
    name = f"{phase}-{experiment_version}-ctx196608"
    assert ns["phase_settings"](phase) == (name, 196608)
    assert ns["phase_run_dir"](phase) == Path(f"runs/private/qwen-{name}")
    config = ns["configured_phase"](phase)
    expected_protocol = "protocol-v1" if phase == "test" else f"development-{experiment_version}"
    assert config.protocol_version == expected_protocol
    assert config.structured_summary_mode == (
        "schema_citations_compact_v1" if experiment_version == "summary-v6" else
        "schema_citations_bounded_v1" if experiment_version == "summary-v5" else
        "schema_citations_v1" if experiment_version in {"summary-v3", "summary-v4"} else "prompt"
    )
    assert config.token_maximum == (
        2048 if experiment_version in {"summary-v4", "summary-v5", "summary-v6"} else 1024
    )
    assert config.summary_max_tokens == (
        3200 if experiment_version in {"summary-v4", "summary-v5", "summary-v6"} else 1600
    )
    assert config.token_fraction == 0.25
    assert config.token_minimum == (
        1024 if experiment_version in {"summary-v5", "summary-v6"} else 128
    )
    assert config.monitor_output_mode == (
        "schema_visible_evidence_v1" if experiment_version == "summary-v6" else "prompt"
    )
    assert config.max_attempts == 2
    assert config.monitor_max_tokens == 700
    assert config.run_dir == f"runs/private/qwen-{name}"
    assert config.split == ("test" if phase == "test" else "development")
    assert config.dataset_dir == "data/private"
    assert config.qwen.model_revision == "b" * 40
    assert config.qwen.gpu_budget_hours == 12
    assert config.qwen.max_model_len == 196608
    assert config.monitor_context_window == config.summarizer_context_window == 196608
    assert config.pilot_pairs == (3 if phase == "pilot" else None)
    saved = root / "versions" / experiment_version / "configuration" / f"{name}.json"
    assert json.loads(saved.read_text()) == config.model_dump()
    assert not (root / "configuration" / f"{name}.json").exists()
    for relative, contents in old_files.items():
        assert (root / relative).read_bytes() == contents
    assert ns["configured_phase"](phase).model_dump() == config.model_dump()


def test_version_reports_and_diagnostics_use_new_run_and_leave_old_results(
    previous_study, monkeypatch, experiment_version,
):
    import subprocess

    root = previous_study
    ns = bootstrap(root, EXPERIMENT_VERSION=experiment_version)
    ns["prepare_version_workspace"]()
    monkeypatch.chdir(root)
    (root / "runs").mkdir()
    # Colab's local checkout can bind private directories; the Drive version itself cannot.
    (root / "runs/private").symlink_to(root / "runs-private", target_is_directory=True)
    current = root / "runs-private" / f"qwen-pilot-{experiment_version}-ctx196608"
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
    numeric = root / "numeric-results" / experiment_version / "pilot"
    assert result["scores"] == str(numeric / "public_scores.csv")
    assert result["report"] == str(numeric / "reproduced/findings.md")
    assert old_report.read_bytes() == old_contents


def test_source_setup_separates_git_and_receipts_but_binds_shared_private_data(
    previous_study, experiment_version,
):
    root = previous_study
    ns = bootstrap(root, EXPERIMENT_VERSION=experiment_version)
    version = root / "versions" / experiment_version
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
        "frozen": False, "run_root": root / "runs-private", "run_version": experiment_version,
    }
    assert (ns["REPO"] / "data/manifests/inventory.json").read_bytes() == (
        version / "public-manifests/inventory.json"
    ).read_bytes()


def test_saving_public_manifests_cannot_overwrite_parent_inventory(
    previous_study, experiment_version,
):
    root = previous_study
    ns = bootstrap(root, EXPERIMENT_VERSION=experiment_version)
    ns["prepare_version_workspace"]()
    original = (root / "public-manifests/inventory.json").read_bytes()
    source = ns["REPO"] / "data/manifests/inventory.json"
    write_json(source, {"synthetic_new_version": True})
    ns["save_public_manifests"]()
    assert (root / "public-manifests/inventory.json").read_bytes() == original
    saved = root / "versions" / experiment_version / "public-manifests/inventory.json"
    assert saved.read_bytes() == source.read_bytes()


def test_v3_inherits_v2_choices_once_and_keeps_all_older_records(previous_study):
    root = previous_study
    v2 = bootstrap(root, EXPERIMENT_VERSION="summary-v2")
    v2["prepare_version_workspace"]()
    parent = v2["source_workspace"]()
    write_json(parent / "configuration/code-pin.json", {"commit": "c" * 40})
    write_json(parent / "configuration/context/selection.json", {"context_window": 229376})
    write_json(parent / "public-manifests/inventory.json", {"synthetic_version_two": True})
    write_json(root / "runs-private/qwen-pilot-summary-v2-ctx196608/manifests/run.json", {
        "config": {"split": "development"}, "run_id": "synthetic-v2",
    })
    write_json(root / "numeric-results/summary-v2/pilot/metrics.json", {"old_result": True})
    before = snapshot(root)
    ns = bootstrap(root, EXPERIMENT_VERSION="summary-v3")
    ns["prepare_version_workspace"]()
    version = ns["source_workspace"]()
    marker = json.loads((version / "configuration/version.json").read_text())
    assert marker["parent"] == "summary-v2"
    assert ns["phase_settings"]("pilot") == ("pilot-summary-v3-ctx229376", 229376)
    for relative in ("configuration/code-pin.json", "configuration/context/selection.json",
                     "public-manifests/inventory.json"):
        assert (version / relative).read_bytes() == (parent / relative).read_bytes()
    for relative, contents in before.items():
        assert (root / relative).read_bytes() == contents
    assert not (version / "runs-private").exists()
    assert not (version / "configuration/model-pin.json").exists()
    assert ns["configured_phase"]("pilot").qwen.model_revision == "b" * 40
    inherited = snapshot(version)
    write_json(parent / "configuration/context/selection.json", {"context_window": 262144})
    write_json(parent / "configuration/code-pin.json", {"commit": "d" * 40})
    resumed = bootstrap(root, EXPERIMENT_VERSION="summary-v3")
    resumed["prepare_version_workspace"]()
    assert snapshot(version) == inherited
    assert resumed["phase_settings"]("pilot") == ("pilot-summary-v3-ctx229376", 229376)


@pytest.mark.parametrize("marker", [None, {"experiment_version": "not-v2"}])
def test_v3_without_valid_v2_marker_inherits_legacy(previous_study, marker):
    root = previous_study
    v2 = root / "versions/summary-v2"
    write_json(v2 / "configuration/context/selection.json", {"context_window": 229376})
    if marker is not None:
        write_json(v2 / "configuration/version.json", marker)
    ns = bootstrap(root, EXPERIMENT_VERSION="summary-v3")
    ns["prepare_version_workspace"]()
    actual = json.loads((ns["source_workspace"]() / "configuration/version.json").read_text())
    assert actual["parent"] == "legacy"
    assert ns["phase_settings"]("pilot") == ("pilot-summary-v3-ctx196608", 196608)


@pytest.mark.parametrize("marker", [
    "versions/summary-v2/frozen-source.zip",
    "versions/summary-v2/public-manifests/protocol-v1.json",
    "runs-private/qwen-test-summary-v2-ctx196608/manifests/run.json",
    "runs-private/custom-test-attempt/manifests/run.json",
])
def test_v3_rejects_any_older_freeze_or_test_evidence(previous_study, marker):
    root = previous_study
    write_json(root / marker, {"config": {"split": "test"}})
    before = snapshot(root)
    ns = bootstrap(root, EXPERIMENT_VERSION="summary-v3")
    with pytest.raises(ValueError, match="frozen|test|review"):
        ns["prepare_version_workspace"]()
    assert snapshot(root) == before


def test_v3_workflow_status_has_its_own_directory(previous_study):
    root = previous_study
    write_json(root / "runs-private/notebook-status/summary-v2/latest.json", {"old_status": True})
    before = snapshot(root)
    ns = bootstrap(root, EXPERIMENT_VERSION="summary-v3")
    ns["record_status"]({"status": "synthetic_no_generation"})
    assert ns["status_directory"]() == root / "runs-private/notebook-status/summary-v3"
    for relative, contents in before.items():
        assert (root / relative).read_bytes() == contents


@pytest.mark.parametrize("version", [
    "legacy", "summary-v2", "summary-v3", "summary-v4", "summary-v5", "summary-v6",
])
def test_schema_versions_pin_the_structured_decoder_without_changing_runtime_pin(
    previous_study, version,
):
    ns = bootstrap(previous_study, EXPERIMENT_VERSION=version)
    pin_path = previous_study / "configuration/model-pin.json"
    before = pin_path.read_bytes()
    engine, _ = ns["install_commands"](json.loads(before))
    assert ("xgrammar==0.2.3" in engine) is (
        version in {"summary-v3", "summary-v4", "summary-v5", "summary-v6"}
    )
    assert "vllm==0.28.0" in engine
    assert pin_path.read_bytes() == before


def test_v4_inherits_v3_choices_once_and_keeps_all_older_records(previous_study):
    root = previous_study
    for version in ("summary-v2", "summary-v3"):
        prior = bootstrap(root, EXPERIMENT_VERSION=version)
        prior["prepare_version_workspace"]()
        write_json(root / f"runs-private/qwen-pilot-{version}-ctx196608/manifests/run.json", {
            "config": {"split": "development"}, "run_id": f"synthetic-{version}",
        })
        write_json(root / f"numeric-results/{version}/pilot/metrics.json", {"old_result": True})
    parent = root / "versions/summary-v3"
    write_json(parent / "configuration/code-pin.json", {"commit": "c" * 40})
    write_json(parent / "public-manifests/inventory.json", {"synthetic_version_three": True})
    before = snapshot(root)
    ns = bootstrap(root, EXPERIMENT_VERSION="summary-v4")
    ns["prepare_version_workspace"]()
    version = ns["source_workspace"]()
    marker = json.loads((version / "configuration/version.json").read_text())
    assert marker["parent"] == "summary-v3"
    assert marker["experiment_version"] == "summary-v4"
    assert "2048" in marker["reason"]
    assert ns["phase_settings"]("pilot") == ("pilot-summary-v4-ctx196608", 196608)
    for relative in ("configuration/code-pin.json", "configuration/context/selection.json",
                     "public-manifests/inventory.json"):
        assert (version / relative).read_bytes() == (parent / relative).read_bytes()
    for relative, contents in before.items():
        assert (root / relative).read_bytes() == contents
    assert not (root / "runs-private/qwen-pilot-summary-v4-ctx196608").exists()
    assert not (version / "configuration/model-pin.json").exists()
    assert ns["configured_phase"]("pilot").qwen.model_revision == "b" * 40
    inherited = snapshot(version)
    write_json(parent / "configuration/context/selection.json", {"context_window": 262144})
    write_json(parent / "configuration/code-pin.json", {"commit": "d" * 40})
    resumed = bootstrap(root, EXPERIMENT_VERSION="summary-v4")
    resumed["prepare_version_workspace"]()
    assert snapshot(version) == inherited
    assert resumed["phase_settings"]("pilot") == ("pilot-summary-v4-ctx196608", 196608)


@pytest.mark.parametrize("v2_valid, v3_valid, expected_parent", [
    (True, False, "summary-v2"), (False, False, "legacy"), (False, True, "summary-v3"),
])
def test_v4_uses_newest_valid_parent(previous_study, v2_valid, v3_valid, expected_parent):
    root = previous_study
    for version, valid in (("summary-v2", v2_valid), ("summary-v3", v3_valid)):
        parent = root / "versions" / version
        write_json(parent / "configuration/version.json", {
            "experiment_version": version if valid else "unrecognized-version",
        })
        write_json(parent / "configuration/context/selection.json", {"context_window": 229376})
    ns = bootstrap(root, EXPERIMENT_VERSION="summary-v4")
    ns["prepare_version_workspace"]()
    actual = json.loads((ns["source_workspace"]() / "configuration/version.json").read_text())
    assert actual["parent"] == expected_parent
    expected_window = 196608 if expected_parent == "legacy" else 229376
    assert ns["phase_settings"]("pilot")[1] == expected_window


@pytest.mark.parametrize("marker", [
    "versions/summary-v2/frozen-source.zip",
    "versions/summary-v2/public-manifests/protocol-v1.json",
    "versions/summary-v3/frozen-source.zip",
    "versions/summary-v3/public-manifests/protocol-v1.json",
    "runs-private/qwen-test-summary-v2-ctx196608/manifests/run.json",
    "runs-private/qwen-test-summary-v3-ctx196608/manifests/run.json",
    "runs-private/custom-test-attempt/manifests/run.json",
])
def test_v4_rejects_any_older_freeze_or_test_evidence(previous_study, marker):
    root = previous_study
    write_json(root / marker, {"config": {"split": "test"}})
    before = snapshot(root)
    ns = bootstrap(root, EXPERIMENT_VERSION="summary-v4")
    with pytest.raises(ValueError, match="Prior frozen/test evidence"):
        ns["prepare_version_workspace"]()
    assert snapshot(root) == before


def test_legacy_configuration_retains_original_summary_allowances(previous_study):
    config = bootstrap(previous_study)["configured_phase"]("development")
    assert config.token_maximum == 1024
    assert config.summary_max_tokens == 1600
    assert config.structured_summary_mode == "prompt"


def test_v5_inherits_v4_choices_once_and_preserves_prior_artifacts_and_budget(previous_study):
    root = previous_study
    for version in ("summary-v2", "summary-v3", "summary-v4"):
        prior = bootstrap(root, EXPERIMENT_VERSION=version)
        prior["prepare_version_workspace"]()
        prior["configured_phase"]("pilot")
        write_json(root / f"runs-private/qwen-pilot-{version}-ctx196608/manifests/run.json", {
            "config": {"split": "development"}, "run_id": f"synthetic-{version}",
        })
        write_json(root / f"numeric-results/{version}/pilot/metrics.json", {"old_result": True})
        write_json(root / f"runs-private/notebook-status/{version}/latest.json", {
            "status": "synthetic_prior_status",
        })
    parent = root / "versions/summary-v4"
    write_json(parent / "configuration/code-pin.json", {"commit": "c" * 40})
    write_json(parent / "configuration/context/selection.json", {"context_window": 229376})
    write_json(parent / "public-manifests/families.json", {"families": ["synthetic-family"]})
    before = snapshot(root)
    ns = bootstrap(root, EXPERIMENT_VERSION="summary-v5")
    before_budget = ns["read_budget"](root)
    ns["prepare_version_workspace"]()
    version = ns["source_workspace"]()
    marker = json.loads((version / "configuration/version.json").read_text())
    assert marker["parent"] == "summary-v4"
    assert marker["experiment_version"] == "summary-v5"
    assert ns["phase_settings"]("pilot") == ("pilot-summary-v5-ctx229376", 229376)
    for relative in ("configuration/code-pin.json", "configuration/context/selection.json",
                     "public-manifests/inventory.json", "public-manifests/families.json"):
        assert (version / relative).read_bytes() == (parent / relative).read_bytes()
    assert not (root / "runs-private/qwen-pilot-summary-v5-ctx229376").exists()
    assert not (version / "configuration/model-pin.json").exists()
    assert ns["configured_phase"]("pilot").qwen.model_revision == "b" * 40
    ns["record_status"]({"status": "synthetic_no_generation"})
    assert ns["status_directory"]() == root / "runs-private/notebook-status/summary-v5"
    for relative, contents in before.items():
        assert (root / relative).read_bytes() == contents
    assert ns["read_budget"](root) == before_budget
    inherited = snapshot(version)
    write_json(parent / "configuration/context/selection.json", {"context_window": 262144})
    write_json(parent / "configuration/code-pin.json", {"commit": "d" * 40})
    resumed = bootstrap(root, EXPERIMENT_VERSION="summary-v5")
    resumed["prepare_version_workspace"]()
    assert snapshot(version) == inherited
    assert resumed["phase_settings"]("pilot") == ("pilot-summary-v5-ctx229376", 229376)


@pytest.mark.parametrize("valid_versions, expected_parent, expected_window", [
    (("summary-v2", "summary-v3", "summary-v4"), "summary-v4", 229376),
    (("summary-v2", "summary-v3"), "summary-v3", 212992),
    (("summary-v2",), "summary-v2", 180224),
    ((), "legacy", 196608),
])
def test_v5_uses_newest_valid_parent(
    previous_study, valid_versions, expected_parent, expected_window,
):
    root = previous_study
    for version, window in (("summary-v2", 180224), ("summary-v3", 212992),
                            ("summary-v4", 229376)):
        parent = root / "versions" / version
        write_json(parent / "configuration/version.json", {
            "experiment_version": version if version in valid_versions else "unrecognized-version",
        })
        write_json(parent / "configuration/context/selection.json", {"context_window": window})
    ns = bootstrap(root, EXPERIMENT_VERSION="summary-v5")
    ns["prepare_version_workspace"]()
    actual = json.loads((ns["source_workspace"]() / "configuration/version.json").read_text())
    assert actual["parent"] == expected_parent
    assert ns["phase_settings"]("pilot")[1] == expected_window


@pytest.mark.parametrize("marker", [
    "versions/summary-v2/frozen-source.zip",
    "versions/summary-v2/public-manifests/protocol-v1.json",
    "versions/summary-v3/frozen-source.zip",
    "versions/summary-v3/public-manifests/protocol-v1.json",
    "versions/summary-v4/frozen-source.zip",
    "versions/summary-v4/public-manifests/protocol-v1.json",
    "runs-private/qwen-test-summary-v2-ctx196608/manifests/run.json",
    "runs-private/qwen-test-summary-v3-ctx196608/manifests/run.json",
    "runs-private/qwen-test-summary-v4-ctx196608/manifests/run.json",
    "runs-private/custom-test-attempt/manifests/run.json",
])
def test_v5_rejects_any_older_freeze_or_test_evidence(previous_study, marker):
    root = previous_study
    write_json(root / marker, {"config": {"split": "test"}})
    before = snapshot(root)
    ns = bootstrap(root, EXPERIMENT_VERSION="summary-v5")
    with pytest.raises(ValueError, match="Prior frozen/test evidence"):
        ns["prepare_version_workspace"]()
    assert snapshot(root) == before


@pytest.mark.parametrize("valid_versions, expected_parent, expected_window", [
    (("summary-v2", "summary-v3", "summary-v4", "summary-v5"), "summary-v5", 245760),
    (("summary-v2", "summary-v3", "summary-v4"), "summary-v4", 229376),
    (("summary-v2", "summary-v3"), "summary-v3", 212992),
    (("summary-v2",), "summary-v2", 180224),
    ((), "legacy", 196608),
])
def test_v6_uses_newest_valid_parent_without_changing_prior_artifacts(
    previous_study, valid_versions, expected_parent, expected_window,
):
    root = previous_study
    for version, window in (("summary-v2", 180224), ("summary-v3", 212992),
                            ("summary-v4", 229376), ("summary-v5", 245760)):
        parent = root / "versions" / version
        write_json(parent / "configuration/version.json", {
            "experiment_version": version if version in valid_versions else "unrecognized-version",
        })
        write_json(parent / "configuration/code-pin.json", {"commit": "f" * 40})
        write_json(parent / "configuration/context/selection.json", {"context_window": window})
        write_json(parent / "public-manifests/families.json", {"synthetic_family": version})
        write_json(root / f"runs-private/qwen-pilot-{version}/manifests/run.json", {
            "config": {"split": "development"}, "run_id": f"synthetic-{version}",
        })
        write_json(root / f"runs-private/qwen-pilot-{version}/cache/synthetic.json", {
            "synthetic_cache": version,
        })
        write_json(root / f"numeric-results/{version}/pilot/metrics.json", {"old_result": True})
    before = snapshot(root)
    ns = bootstrap(root, EXPERIMENT_VERSION="summary-v6")
    budget = ns["read_budget"](root)
    ns["prepare_version_workspace"]()
    version = ns["source_workspace"]()
    actual = json.loads((version / "configuration/version.json").read_text())
    assert actual["parent"] == expected_parent
    assert ns["phase_settings"]("pilot") == (f"pilot-summary-v6-ctx{expected_window}", expected_window)
    parent = root if expected_parent == "legacy" else root / "versions" / expected_parent
    for relative in ("configuration/code-pin.json", "configuration/context/selection.json"):
        assert (version / relative).read_bytes() == (parent / relative).read_bytes()
    assert not (version / "configuration/model-pin.json").exists()
    assert not (root / f"runs-private/qwen-pilot-summary-v6-ctx{expected_window}").exists()
    assert ns["read_budget"](root) == budget
    for relative, contents in before.items():
        assert (root / relative).read_bytes() == contents


@pytest.mark.parametrize("marker", [
    "versions/summary-v2/frozen-source.zip",
    "versions/summary-v2/public-manifests/protocol-v1.json",
    "versions/summary-v3/frozen-source.zip",
    "versions/summary-v3/public-manifests/protocol-v1.json",
    "versions/summary-v4/frozen-source.zip",
    "versions/summary-v4/public-manifests/protocol-v1.json",
    "versions/summary-v5/frozen-source.zip",
    "versions/summary-v5/public-manifests/protocol-v1.json",
    "runs-private/qwen-test-summary-v2-ctx196608/manifests/run.json",
    "runs-private/qwen-test-summary-v3-ctx196608/manifests/run.json",
    "runs-private/qwen-test-summary-v4-ctx196608/manifests/run.json",
    "runs-private/qwen-test-summary-v5-ctx196608/manifests/run.json",
    "runs-private/custom-test-attempt/manifests/run.json",
])
def test_v6_rejects_any_older_freeze_or_test_evidence(previous_study, marker):
    root = previous_study
    write_json(root / marker, {"config": {"split": "test"}})
    before = snapshot(root)
    ns = bootstrap(root, EXPERIMENT_VERSION="summary-v6")
    with pytest.raises(ValueError, match="Prior frozen/test evidence"):
        ns["prepare_version_workspace"]()
    assert snapshot(root) == before


def test_v5_reconnect_accepts_saved_configuration_before_monitor_mode_field(previous_study):
    ns = bootstrap(previous_study, EXPERIMENT_VERSION="summary-v5")
    ns["prepare_version_workspace"]()
    config = ns["configured_phase"]("pilot")
    path = ns["source_workspace"]() / "configuration/pilot-summary-v5-ctx196608.json"
    saved = config.model_dump()
    saved.pop("monitor_output_mode", None)
    write_json(path, saved)
    before = path.read_bytes()
    restored = ns["configured_phase"]("pilot")
    assert restored.monitor_output_mode == "prompt"
    assert restored.structured_summary_mode == "schema_citations_bounded_v1"
    assert path.read_bytes() == before


def test_v6_reconnect_rejects_changed_monitor_method_without_rewriting_config(previous_study):
    ns = bootstrap(previous_study, EXPERIMENT_VERSION="summary-v6")
    ns["prepare_version_workspace"]()
    config = ns["configured_phase"]("pilot")
    path = ns["source_workspace"]() / "configuration/pilot-summary-v6-ctx196608.json"
    saved = config.model_dump()
    saved["monitor_output_mode"] = "prompt"
    write_json(path, saved)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="different saved configuration"):
        ns["configured_phase"]("pilot")
    assert path.read_bytes() == before
