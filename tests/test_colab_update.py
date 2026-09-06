"""Source migration preserves pinned provenance and unrelated local edits."""

import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, text=True, capture_output=True
    ).stdout.strip()


@pytest.fixture
def source(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Synthetic migration test")
    _git(repo, "config", "user.email", "fixture@invalid")
    (repo / ".gitignore").write_text("runs/private\ndata/private\ndist/\n")
    code = repo / "src/context_audit/colab.py"
    code.parent.mkdir(parents=True)
    code.write_text("BASE = 'independent synthetic fixture'\n")
    (repo / "README.md").write_text("Independent source fixture.\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "Independent synthetic baseline")
    clone = tmp_path / "target"
    subprocess.run(["git", "clone", "--quiet", str(repo), str(clone)], check=True)
    code.write_text("BASE = 'updated independent synthetic fixture'\n")
    return repo, clone


def test_update_preserves_head_private_data_and_unrelated_changes(source, tmp_path):
    repo, target = source
    build = _module("build_colab_update")
    apply = _module("apply_colab_update")
    archive = tmp_path / "update.zip"
    build.build_update(repo, archive, compatibility_patch=None)
    head = _git(target, "rev-parse", "HEAD")
    (target / "README.md").write_text("Unrelated local work.\n")
    private = target / "data/private/independent.txt"
    private.parent.mkdir(parents=True)
    private.write_text("Independent private sentinel.\n")
    durable_runs = tmp_path / "drive-private-runs"
    durable_runs.mkdir()
    (target / "runs").mkdir()
    (target / "runs/private").symlink_to(durable_runs, target_is_directory=True)
    result = apply.apply_update(target, archive)
    assert result["updated_files"] == ["src/context_audit/colab.py"]
    assert _git(target, "rev-parse", "HEAD") == head
    assert private.read_text() == "Independent private sentinel.\n"
    assert (target / "README.md").read_text() == "Unrelated local work.\n"
    assert (target / "src/context_audit/colab.py").read_bytes() == (
        repo / "src/context_audit/colab.py"
    ).read_bytes()
    assert Path(result["receipt"]).is_file()
    assert Path(result["receipt"]).resolve().is_relative_to(durable_runs)
    assert apply.apply_update(target, archive)["updated_files"] == []


def test_update_rejects_conflicting_edits_before_writing_any_file(source, tmp_path):
    repo, target = source
    (repo / "README.md").write_text("Updated documentation.\n")
    archive = tmp_path / "update.zip"
    _module("build_colab_update").build_update(repo, archive, compatibility_patch=None)
    code = target / "src/context_audit/colab.py"
    code.write_text("LOCAL = 'preserve this independent edit'\n")
    with pytest.raises(ValueError, match="modified|hash"):
        _module("apply_colab_update").apply_update(target, archive)
    assert (target / "README.md").read_text() == "Independent source fixture.\n"
    assert code.read_text() == "LOCAL = 'preserve this independent edit'\n"
    assert not (target / "runs/private").exists()


def test_update_rejects_different_git_head(source, tmp_path):
    repo, target = source
    archive = tmp_path / "update.zip"
    _module("build_colab_update").build_update(repo, archive, compatibility_patch=None)
    _git(target, "-c", "user.name=Fixture", "-c", "user.email=fixture@invalid", "commit",
         "--allow-empty", "--quiet", "-m", "Another source revision")
    with pytest.raises(ValueError, match="HEAD|commit"):
        _module("apply_colab_update").apply_update(target, archive)


@pytest.mark.parametrize("unsafe", ["../escape.py", "data/private/source.py", ".git/config"])
def test_update_rejects_nonpublic_paths(source, tmp_path, unsafe):
    repo, target = source
    archive = tmp_path / "update.zip"
    _module("build_colab_update").build_update(repo, archive, compatibility_patch=None)
    with zipfile.ZipFile(archive) as original:
        contents = {name: original.read(name) for name in original.namelist()}
    manifest = json.loads(contents["manifest.json"])
    manifest["files"][0]["path"] = unsafe
    contents["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(archive, "w") as modified:
        for name, body in contents.items():
            modified.writestr(name, body)
    with pytest.raises(ValueError, match="public|path"):
        _module("apply_colab_update").apply_update(target, archive)


def test_update_rejects_source_symlink_escape(source, tmp_path):
    repo, target = source
    archive = tmp_path / "update.zip"
    _module("build_colab_update").build_update(repo, archive, compatibility_patch=None)
    code = target / "src/context_audit/colab.py"
    outside = tmp_path / "outside.py"
    outside.write_bytes(code.read_bytes())
    code.unlink()
    code.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink|escape"):
        _module("apply_colab_update").apply_update(target, archive)
    assert outside.read_text() == "BASE = 'independent synthetic fixture'\n"


def test_update_accepts_reviewed_intermediate_hardware_patch(source, tmp_path, monkeypatch):
    import hashlib

    repo, target = source
    code = repo / "src/context_audit/colab.py"
    final = code.read_bytes()
    code.write_text("BASE = 'independent hardware compatibility fixture'\n")
    patch = tmp_path / "hardware.patch"
    patch.write_text(_git(repo, "diff", "--", "src/context_audit/colab.py") + "\n")
    code.write_bytes(final)
    build = _module("build_colab_update")
    monkeypatch.setattr(
        build, "HARDWARE_PATCH_SHA256", hashlib.sha256(patch.read_bytes()).hexdigest()
    )
    archive = tmp_path / "update.zip"
    build.build_update(repo, archive, compatibility_patch=patch)
    _git(target, "apply", str(patch))
    _module("apply_colab_update").apply_update(target, archive)
    assert (target / "src/context_audit/colab.py").read_bytes() == final


def test_update_adds_new_public_source_without_overwriting_existing_file(source, tmp_path):
    repo, target = source
    new = repo / "src/context_audit/synthetic_new.py"
    new.write_text("NEW = 'independent synthetic fixture'\n")
    archive = tmp_path / "update.zip"
    _module("build_colab_update").build_update(repo, archive, compatibility_patch=None)
    collision = target / "src/context_audit/synthetic_new.py"
    collision.write_text("EXISTING = 'untracked user edit'\n")
    with pytest.raises(ValueError, match="modified|hash"):
        _module("apply_colab_update").apply_update(target, archive)
    assert collision.read_text() == "EXISTING = 'untracked user edit'\n"
    collision.unlink()
    _module("apply_colab_update").apply_update(target, archive)
    assert collision.read_bytes() == new.read_bytes()


def test_update_cell_preserves_setup_and_review_flags_without_generation(
    source, tmp_path, monkeypatch
):
    import sys
    from types import ModuleType, SimpleNamespace

    repo, target = source
    notebook = repo / "notebooks/03_qwen_colab.ipynb"
    notebook.parent.mkdir()
    notebook.write_text(json.dumps({"cells": [
        {"cell_type": "code", "source": ["def configured_phase(phase): return phase\n"]},
        {"cell_type": "code", "source": [
            "# LIVE_EXECUTION\nraise RuntimeError('model invoked')\n"
        ]},
    ]}))
    _git(repo, "add", "notebooks/03_qwen_colab.ipynb")
    _git(repo, "commit", "--quiet", "-m", "Independent notebook baseline")
    _git(target, "pull", "--quiet", "--ff-only")
    archive = tmp_path / "update.zip"
    result = _module("build_colab_update").build_update(repo, archive, compatibility_patch=None)
    colab = ModuleType("google.colab")
    colab.files = SimpleNamespace(upload=lambda: {"update.zip": archive.read_bytes()})
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    namespace = {
        "REPO": target, "SETUP_READY": True, "DATA_USE_CONFIRMED": False,
        "RUBRIC_REVIEWED": True, "PHASE": "pilot", "RUN_LIVE": True,
    }
    cell = Path(result["update_cell"]).read_text()
    original_modules = {
        name: module for name, module in sys.modules.items()
        if name == "context_audit" or name.startswith("context_audit.")
    }
    try:
        exec(compile(cell, "independent-colab-update", "exec"), namespace)
    finally:
        sys.modules.update(original_modules)
    assert namespace["SETUP_READY"] is True
    assert namespace["DATA_USE_CONFIRMED"] is False
    assert namespace["RUBRIC_REVIEWED"] is True
    assert namespace["RUN_LIVE"] is False
    assert namespace["MAX_GPU_HOURS"] == 12.0
    assert namespace["configured_phase"]("pilot") == "pilot"
    with pytest.raises(ValueError, match="RUN_LIVE"):
        namespace["run_updated_pilot"]()
