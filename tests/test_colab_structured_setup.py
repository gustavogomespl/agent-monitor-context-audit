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
        assert command[-2:] == ['-m', 'context_audit.structured_backend']
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
        receipt["structured_summary_mode"] = "schema_citations_separate_ids_v1"

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
    assert "DECODER_LATENCY_OK" in capsys.readouterr().out
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
        receipt["structured_summary_mode"] = "schema_citations_separate_ids_v1"

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
