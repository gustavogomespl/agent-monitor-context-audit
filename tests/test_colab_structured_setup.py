"""A fresh CPU subprocess checks the grammar before the guided model startup."""

import json
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/colab_bootstrap.py'


def bootstrap(tmp_path, version):
    ns = runpy.run_path(str(SCRIPT))['install_runtime'].__globals__
    ns.update(EXPERIMENT_VERSION=version, REPO=tmp_path)
    return ns


@pytest.mark.parametrize('version', ['legacy', 'summary-v2'])
def test_old_versions_do_not_run_new_grammar_probe(tmp_path, monkeypatch, version):
    ns = bootstrap(tmp_path, version)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('Unexpected probe'))
    assert ns['prepare_structured_outputs']() == {'status': 'not_requested'}


@pytest.mark.parametrize('version', [
    'summary-v3', 'summary-v4', 'summary-v5', 'summary-v6', 'summary-v7',
])
def test_probe_runs_before_any_model_load_in_fresh_process(tmp_path, monkeypatch, capsys, version):
    ns = bootstrap(tmp_path, version)
    receipt = {'status': 'passed', 'backend': 'xgrammar', 'version': '0.2.3',
               'accepted_cases': 2, 'rejected_cases': 41, 'model_generation_executed': False}

    def run(command, **kwargs):
        expected = ['-m', 'context_audit.structured_backend']
        if version == 'summary-v7':
            expected.append('--separate-ids')
        assert command[1:] == expected
        assert kwargs['cwd'] == tmp_path and kwargs['timeout'] <= 120
        assert '--model' not in command
        return SimpleNamespace(returncode=0, stdout=json.dumps(receipt), stderr='')

    monkeypatch.setattr(subprocess, 'run', run)
    assert ns['prepare_structured_outputs']() == receipt
    assert 'STRUCTURED_OUTPUTS_OK' in capsys.readouterr().out


@pytest.mark.parametrize('version', [
    'summary-v3', 'summary-v4', 'summary-v5', 'summary-v6', 'summary-v7',
])
def test_probe_failure_stops_setup_without_unconstrained_fallback(tmp_path, monkeypatch, version):
    ns = bootstrap(tmp_path, version)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1, stdout='', stderr='Synthetic compiler rejected schema')

    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='Synthetic compiler rejected schema'):
        ns['prepare_structured_outputs']()
    assert len(calls) == 1


@pytest.mark.parametrize("version", [
    "legacy", "summary-v2", "summary-v3", "summary-v4", "summary-v5",
])
def test_older_versions_skip_full_vocabulary_latency_probe(tmp_path, monkeypatch, version):
    ns = bootstrap(tmp_path, version)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Unexpected probe"))
    assert ns["prepare_decoder_latency"]() == {"status": "not_requested"}


@pytest.mark.parametrize("version", ["summary-v6", "summary-v7"])
def test_latency_probe_uses_pinned_tokenizer_and_persists_receipt(
    tmp_path, monkeypatch, capsys, version,
):
    import sys

    ns = bootstrap(tmp_path, version)
    ns["DRIVE_ROOT"] = tmp_path / "drive"
    pin = ns["DRIVE_ROOT"] / "configuration/model-pin.json"
    pin.parent.mkdir(parents=True)
    pin.write_text(json.dumps({"model_id": "Qwen/Qwen3.8-27B", "model_revision": "b" * 40}))
    receipt = {"status": "passed", "model_generation_executed": False,
               "vocab_size": 300000, "max_mask_seconds": 0.025}
    if version == "summary-v7":
        receipt.update(
            structured_summary_mode="schema_citations_separate_ids_v1",
            mask_gate_policy="two_fast_confirmations_v1",
            max_mask_seconds=0.4, max_gate_mask_seconds=0.025, mask_gate_seconds=0.25,
        )

    def run(command, **kwargs):
        expected = [sys.executable, "-m", "context_audit.decoder_latency",
                           "--model", "Qwen/Qwen3.8-27B", "--revision", "b" * 40]
        if version == "summary-v7":
            expected.extend(["--structured-summary-mode", "schema_citations_separate_ids_v1"])
        assert command == expected
        assert kwargs["cwd"] == tmp_path
        assert kwargs["timeout"] == 180
        return SimpleNamespace(returncode=0, stdout=json.dumps(receipt), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    assert ns["prepare_decoder_latency"]() == receipt
    saved = ns["DRIVE_ROOT"] / f"versions/{version}/configuration/decoder-latency.json"
    assert json.loads(saved.read_text()) == receipt
    output = capsys.readouterr().out
    assert "DECODER_LATENCY_OK" in output
    if version == "summary-v7":
        assert "raw maximum mask seconds: 0.4" in output
        assert "confirmed gate maximum mask seconds: 0.025" in output
    else:
        assert "maximum mask seconds: 0.025" in output
        assert "confirmed" not in output and "raw maximum" not in output
    assert ns["SETUP_READY"] is False


@pytest.mark.parametrize("result", [
    SimpleNamespace(returncode=1, stdout="", stderr="Synthetic mask latency exceeded limit"),
    SimpleNamespace(returncode=0, stdout='{"status": "failed"}', stderr=""),
    SimpleNamespace(returncode=0, stdout='{"status": "passed", "model_generation_executed": true}',
                    stderr=""),
])
def test_v6_latency_failure_blocks_setup_without_fallback(tmp_path, monkeypatch, result):
    ns = bootstrap(tmp_path, "summary-v6")
    ns["DRIVE_ROOT"] = tmp_path / "drive"
    pin = ns["DRIVE_ROOT"] / "configuration/model-pin.json"
    pin.parent.mkdir(parents=True)
    pin.write_text(json.dumps({"model_id": "Qwen/Qwen3.8-27B", "model_revision": "b" * 40}))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: result)
    with pytest.raises(RuntimeError, match="latency|receipt"):
        ns["prepare_decoder_latency"]()
    assert ns["SETUP_READY"] is False
    path = ns["DRIVE_ROOT"] / "versions/summary-v6/configuration/decoder-latency.json"
    assert not path.exists()


def test_v6_failed_latency_receipt_is_not_hidden_by_import_warnings(tmp_path, monkeypatch):
    ns = bootstrap(tmp_path, "summary-v6")
    ns["DRIVE_ROOT"] = tmp_path / "drive"
    pin = ns["DRIVE_ROOT"] / "configuration/model-pin.json"
    pin.parent.mkdir(parents=True)
    pin.write_text(json.dumps({"model_id": "Qwen/Qwen3.8-27B", "model_revision": "b" * 40}))
    result = SimpleNamespace(
        returncode=1,
        stdout=json.dumps({"status": "failed", "reason_code": "mask_latency",
                           "mask_seconds": 0.3, "model_generation_executed": False}),
        stderr="Synthetic optional import warning",
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: result)
    with pytest.raises(RuntimeError, match="mask_latency"):
        ns["prepare_decoder_latency"]()


@pytest.fixture
def preserve_context_imports():
    """The real setup clears kernel imports; keep that action inside this test."""
    import sys

    original = {
        name: module for name, module in sys.modules.items()
        if name == "context_audit" or name.startswith("context_audit.")
    }
    original_path = list(sys.path)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == "context_audit" or name.startswith("context_audit."):
                del sys.modules[name]
        sys.modules.update(original)
        sys.path[:] = original_path


@pytest.mark.parametrize("version", ["summary-v6", "summary-v7"])
@pytest.mark.parametrize("latency_passed", [True, False])
def test_setup_runs_latency_after_decoder_and_before_ready(
    tmp_path, monkeypatch, latency_passed, preserve_context_imports, version,
):
    import importlib.metadata

    ns = bootstrap(tmp_path, version)
    ns.update(DRIVE_ROOT=tmp_path / "drive", SETUP_READY=True)
    pin = ns["DRIVE_ROOT"] / "configuration/model-pin.json"
    pin.parent.mkdir(parents=True)
    pin.write_text(json.dumps({
        "model_id": "Qwen/Qwen3.8-27B", "model_revision": "b" * 40, "vllm_version": "0.28.0",
    }))
    ns["prepare_version_workspace"]()
    phases = []
    receipt = {"status": "passed", "model_generation_executed": False,
               "vocab_size": 300000, "max_mask_seconds": 0.025}
    if version == "summary-v7":
        receipt.update(
            structured_summary_mode="schema_citations_separate_ids_v1",
            mask_gate_policy="two_fast_confirmations_v1",
            max_mask_seconds=0.4, max_gate_mask_seconds=0.025, mask_gate_seconds=0.25,
        )

    def imports():
        phases.append("imports")
        assert ns["SETUP_READY"] is False
        return {"status": "passed"}

    def run(command, **kwargs):
        assert ns["SETUP_READY"] is False
        if "install" in command:
            phases.append("install")
            return SimpleNamespace(returncode=0)
        if "context_audit.structured_backend" in command:
            assert phases[-1] == "imports"
            phases.append("decoder")
            return SimpleNamespace(returncode=0, stdout=json.dumps({
                "status": "passed", "model_generation_executed": False, "version": "0.2.3",
            }), stderr="")
        if "context_audit.decoder_latency" in command:
            assert phases[-1] == "decoder"
            phases.append("latency")
            return SimpleNamespace(returncode=0 if latency_passed else 1,
                                   stdout=json.dumps(receipt), stderr="Synthetic latency failure")
        assert latency_passed and phases[-1] == "latency"
        assert "freeze" in command or command == ["git", "rev-parse", "HEAD"]
        return SimpleNamespace(returncode=0, stdout="a" * 40, stderr="")

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.0.0")
    monkeypatch.setattr(subprocess, "run", run)
    ns["prepare_vllm_imports"] = imports
    if latency_passed:
        ns["install_runtime"]()
        histories = list((ns["source_workspace"]() / "configuration/setup-history").glob("*.json"))
        assert len(histories) == 1
        assert json.loads(histories[0].read_text())["decoder_latency_check"] == receipt
    else:
        with pytest.raises(RuntimeError, match="Synthetic latency failure"):
            ns["install_runtime"]()
        assert not (ns["source_workspace"]() / "configuration/setup-history").exists()
    assert ns["SETUP_READY"] is latency_passed
    assert phases == ["install", "install", "imports", "decoder", "latency"]


@pytest.mark.parametrize("reported_mode", [None, "schema_citations_compact_v1"])
def test_v7_requires_a_latency_receipt_for_its_production_schema(
    tmp_path, monkeypatch, reported_mode,
):
    ns = bootstrap(tmp_path, "summary-v7")
    ns["DRIVE_ROOT"] = tmp_path / "drive"
    pin = ns["DRIVE_ROOT"] / "configuration/model-pin.json"
    pin.parent.mkdir(parents=True)
    pin.write_text(json.dumps({"model_id": "Qwen/Qwen3.8-27B", "model_revision": "b" * 40}))
    receipt = {"status": "passed", "model_generation_executed": False,
               "vocab_size": 300000, "max_mask_seconds": 0.025}
    if reported_mode is not None:
        receipt["structured_summary_mode"] = reported_mode
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=json.dumps(receipt), stderr="",
    ))
    with pytest.raises(RuntimeError, match="production schema"):
        ns["prepare_decoder_latency"]()
    path = ns["DRIVE_ROOT"] / "versions/summary-v7/configuration/decoder-latency.json"
    assert not path.exists()
    assert ns["SETUP_READY"] is False


@pytest.mark.parametrize("reported_policy", [None, "single_measurement_v1", "unknown_policy"])
def test_v7_requires_the_reviewed_mask_confirmation_policy(tmp_path, monkeypatch, reported_policy):
    ns = bootstrap(tmp_path, "summary-v7")
    ns["DRIVE_ROOT"] = tmp_path / "drive"
    pin = ns["DRIVE_ROOT"] / "configuration/model-pin.json"
    pin.parent.mkdir(parents=True)
    pin.write_text(json.dumps({"model_id": "Qwen/Qwen3.8-27B", "model_revision": "b" * 40}))
    receipt = {"status": "passed", "model_generation_executed": False,
               "structured_summary_mode": "schema_citations_separate_ids_v1",
               "vocab_size": 300000, "max_mask_seconds": 0.4, "max_gate_mask_seconds": 0.025}
    if reported_policy is not None:
        receipt["mask_gate_policy"] = reported_policy
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=json.dumps(receipt), stderr="",
    ))
    with pytest.raises(RuntimeError, match="confirmation policy"):
        ns["prepare_decoder_latency"]()
    path = ns["DRIVE_ROOT"] / "versions/summary-v7/configuration/decoder-latency.json"
    assert not path.exists()
    assert ns["SETUP_READY"] is False


@pytest.mark.parametrize("field,value", [
    ("max_gate_mask_seconds", None),
    ("max_gate_mask_seconds", True),
    ("max_gate_mask_seconds", "0.025"),
    ("max_gate_mask_seconds", float("nan")),
    ("max_gate_mask_seconds", float("inf")),
    ("max_gate_mask_seconds", -0.001),
    ("max_gate_mask_seconds", 0.251),
    ("mask_gate_seconds", None),
    ("mask_gate_seconds", True),
    ("mask_gate_seconds", "0.25"),
    ("mask_gate_seconds", float("nan")),
    ("mask_gate_seconds", 0.5),
])
def test_v7_rejects_unverified_or_slow_confirmed_mask_receipts(tmp_path, monkeypatch, field, value):
    ns = bootstrap(tmp_path, "summary-v7")
    ns["DRIVE_ROOT"] = tmp_path / "drive"
    pin = ns["DRIVE_ROOT"] / "configuration/model-pin.json"
    pin.parent.mkdir(parents=True)
    pin.write_text(json.dumps({"model_id": "Qwen/Qwen3.8-27B", "model_revision": "b" * 40}))
    receipt = {"status": "passed", "model_generation_executed": False,
               "structured_summary_mode": "schema_citations_separate_ids_v1",
               "mask_gate_policy": "two_fast_confirmations_v1",
               "vocab_size": 300000, "max_mask_seconds": 0.4,
               "max_gate_mask_seconds": 0.025, "mask_gate_seconds": 0.25}
    if value is None:
        receipt.pop(field)
    else:
        receipt[field] = value
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=json.dumps(receipt), stderr="",
    ))
    with pytest.raises(RuntimeError, match="confirmed mask timing"):
        ns["prepare_decoder_latency"]()
    path = ns["DRIVE_ROOT"] / "versions/summary-v7/configuration/decoder-latency.json"
    assert not path.exists()
    assert ns["SETUP_READY"] is False


@pytest.mark.parametrize("version,returncode", [
    ("summary-v6", 1), ("summary-v7", 0), ("summary-v7", 1),
])
def test_failed_latency_receipts_preserve_all_v7_timings_without_overwriting_success(
    tmp_path, monkeypatch, version, returncode,
):
    ns = bootstrap(tmp_path, version)
    ns["DRIVE_ROOT"] = tmp_path / "drive"
    pin = ns["DRIVE_ROOT"] / "configuration/model-pin.json"
    pin.parent.mkdir(parents=True)
    pin.write_text(json.dumps({"model_id": "Qwen/Qwen3.8-27B", "model_revision": "b" * 40}))
    configuration = ns["source_workspace"]() / "configuration"
    configuration.mkdir(parents=True)
    success = configuration / "decoder-latency.json"
    success.write_text('{"status":"passed","synthetic_previous_receipt":true}\n')
    previous = success.read_bytes()
    receipt = {"status": "failed", "model_generation_executed": False,
               "reason_code": "mask_latency", "mask_gate_policy": "two_fast_confirmations_v1",
               "measurements": [{"position": i, "mask_seconds": i / 10000} for i in range(1000)]}
    encoded = json.dumps(receipt)
    assert len(encoded) > 12000
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=returncode, stdout=encoded, stderr="Synthetic optional import warning",
    ))
    errors = []
    for _ in range(2):
        with pytest.raises(RuntimeError, match="before model startup") as raised:
            ns["prepare_decoder_latency"]()
        errors.append(str(raised.value))
        assert ns["SETUP_READY"] is False
    failures = sorted((configuration / "decoder-latency-failures").glob("*.json"))
    if version == "summary-v7":
        assert len(failures) == 2 and failures[0] != failures[1]
        assert all(json.loads(path.read_text()) == receipt for path in failures)
        assert all(any(str(path) in error for error in errors) for path in failures)
        assert all("Full private decoder receipt:" in error for error in errors)
    else:
        assert failures == []
        assert all("Full private decoder receipt:" not in error for error in errors)
    assert all(len(error) < 13000 for error in errors)
    assert success.read_bytes() == previous


def test_v7_failure_receipt_write_error_is_not_silently_ignored(tmp_path, monkeypatch):
    ns = bootstrap(tmp_path, "summary-v7")
    ns["DRIVE_ROOT"] = tmp_path / "drive"
    pin = ns["DRIVE_ROOT"] / "configuration/model-pin.json"
    pin.parent.mkdir(parents=True)
    pin.write_text(json.dumps({"model_id": "Qwen/Qwen3.8-27B", "model_revision": "b" * 40}))
    failure_directory = ns["source_workspace"]() / "configuration/decoder-latency-failures"
    failure_directory.parent.mkdir(parents=True)
    failure_directory.write_text("Synthetic obstruction that must be preserved")
    receipt = {"status": "failed", "model_generation_executed": False,
               "reason_code": "mask_latency"}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=1, stdout=json.dumps(receipt), stderr="",
    ))
    with pytest.raises(OSError):
        ns["prepare_decoder_latency"]()
    assert failure_directory.read_text() == "Synthetic obstruction that must be preserved"
    assert ns["SETUP_READY"] is False
