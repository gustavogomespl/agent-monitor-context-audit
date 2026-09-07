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


@pytest.mark.parametrize('version', ['summary-v3', 'summary-v4'])
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


@pytest.mark.parametrize('version', ['summary-v3', 'summary-v4'])
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
