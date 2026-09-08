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


@pytest.mark.parametrize('version', ['summary-v3', 'summary-v4', 'summary-v5', 'summary-v6'])
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


@pytest.mark.parametrize('version', ['summary-v3', 'summary-v4', 'summary-v5', 'summary-v6'])
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


@pytest.mark.parametrize("version", ["legacy", "summary-v2", "summary-v3", "summary-v4", "summary-v5"])
def test_older_versions_skip_full_vocabulary_latency_probe(tmp_path, monkeypatch, version):
    ns = bootstrap(tmp_path, version)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Unexpected probe"))
    assert ns["prepare_decoder_latency"]() == {"status": "not_requested"}


def test_v6_latency_probe_uses_pinned_tokenizer_and_persists_receipt(tmp_path, monkeypatch, capsys):
    import sys

    ns = bootstrap(tmp_path, "summary-v6")
    ns["DRIVE_ROOT"] = tmp_path / "drive"
    pin = ns["DRIVE_ROOT"] / "configuration/model-pin.json"
    pin.parent.mkdir(parents=True)
    pin.write_text(json.dumps({"model_id": "Qwen/Qwen3.8-27B", "model_revision": "b" * 40}))
    receipt = {"status": "passed", "model_generation_executed": False,
               "vocab_size": 300000, "max_mask_seconds": 0.025}

    def run(command, **kwargs):
        assert command == [sys.executable, "-m", "context_audit.decoder_latency",
                           "--model", "Qwen/Qwen3.8-27B", "--revision", "b" * 40]
        assert kwargs["cwd"] == tmp_path
        assert kwargs["timeout"] == 180
        return SimpleNamespace(returncode=0, stdout=json.dumps(receipt), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    assert ns["prepare_decoder_latency"]() == receipt
    saved = ns["DRIVE_ROOT"] / "versions/summary-v6/configuration/decoder-latency.json"
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
    assert not (ns["DRIVE_ROOT"] / "versions/summary-v6/configuration/decoder-latency.json").exists()
