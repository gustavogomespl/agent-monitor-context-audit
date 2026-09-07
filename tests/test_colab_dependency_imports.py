"""CUDA wheel repair and import checks use synthetic package/process boundaries."""

import importlib.metadata
import json
import subprocess
from types import SimpleNamespace

import pytest
from test_colab_notebook import _cells


def namespace():
    result = {}
    for source in _cells():
        exec(source, result)
    return result


@pytest.fixture
def simulated_packages(monkeypatch):
    versions = {"torch": "2.13.0", "vllm": "0.28.0", "torchaudio": "2.11.0+cu128"}
    commands = []
    state = SimpleNamespace(fail_install=False, fail_import=False, cuda="13.0")

    def run(command, **kwargs):
        commands.append(command)
        assert "--model" not in command
        if "torch.version.cuda" in command[-1]:
            return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
                "version": "2.13.0+cu" + state.cuda.replace(".", ""), "cuda": state.cuda,
            }))
        if "pip" in command:
            assert "--no-deps" in command
            assert "torchaudio==2.11.0+cu130" in command
            assert "https://download.pytorch.org/whl/cu130" in command
            assert "--only-binary=:all:" in command
            if state.fail_install:
                raise subprocess.CalledProcessError(1, command)
            versions["torchaudio"] = "2.11.0+cu130"
        else:
            assert "from vllm.entrypoints.openai import api_server" in command[-1]
            assert "import torchaudio" in command[-1]
            assert kwargs["timeout"] > 0
            if state.fail_import:
                return SimpleNamespace(returncode=1, stdout="", stderr="Synthetic ABI mismatch")
        return SimpleNamespace(returncode=0, stdout="VLLM_IMPORT_OK\n", stderr="")

    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(subprocess, "run", run)
    return versions, commands, state


def test_mismatched_audio_build_is_repaired_without_changing_torch(simulated_packages):
    versions, commands, _ = simulated_packages
    ns = namespace()
    assert "prepare_vllm_imports" in ns
    result = ns["prepare_vllm_imports"]()
    assert versions == {"torch": "2.13.0", "vllm": "0.28.0", "torchaudio": "2.11.0+cu130"}
    assert len(commands) == 3
    assert result["before"]["torchaudio"] == "2.11.0+cu128"
    assert result["after"]["torchaudio"] == "2.11.0+cu130"
    assert result["status"] == "passed"


def test_matching_build_only_runs_the_fresh_import_probe(simulated_packages):
    versions, commands, _ = simulated_packages
    versions["torchaudio"] = "2.11.0+cu130"
    ns = namespace()
    assert "prepare_vllm_imports" in ns
    ns["prepare_vllm_imports"]()
    assert len(commands) == 2
    assert not any("pip" in command for command in commands)


def test_import_failure_is_reported_without_retrying_model_startup(simulated_packages):
    _, commands, state = simulated_packages
    state.fail_import = True
    ns = namespace()
    assert "prepare_vllm_imports" in ns
    with pytest.raises(RuntimeError, match="Synthetic ABI mismatch"):
        ns["prepare_vllm_imports"]()
    assert len(commands) == 3


def test_install_failure_never_runs_import_probe(simulated_packages):
    _, commands, state = simulated_packages
    state.fail_install = True
    ns = namespace()
    assert "prepare_vllm_imports" in ns
    with pytest.raises(subprocess.CalledProcessError):
        ns["prepare_vllm_imports"]()
    assert len(commands) == 2


def test_other_torch_build_is_not_replaced_by_the_known_cu130_repair(simulated_packages):
    versions, commands, state = simulated_packages
    versions["torch"] = "2.13.0+cu128"
    state.cuda = "12.8"
    state.fail_import = True
    ns = namespace()
    assert "prepare_vllm_imports" in ns
    with pytest.raises(RuntimeError, match="Synthetic ABI mismatch"):
        ns["prepare_vllm_imports"]()
    assert len(commands) == 2 and "pip" not in commands[0]


def test_failed_repeat_setup_clears_previous_ready_flag(monkeypatch, tmp_path):
    import subprocess
    import urllib.request

    ns = namespace()

    def fail(*args, **kwargs):
        raise RuntimeError("Synthetic dependency setup failure")

    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(urllib.request, "urlopen", fail)
    ns.update(SETUP_READY=True, DRIVE_ROOT=tmp_path)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        ns["install_runtime"]()
    assert ns["SETUP_READY"] is False
