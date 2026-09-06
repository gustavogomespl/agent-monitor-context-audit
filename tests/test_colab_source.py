"""Branch setup must resolve synthetic local source once and preserve that identity."""

import json
import shutil
import subprocess
from pathlib import Path

import nbformat
import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/03_qwen_colab.ipynb"


@pytest.fixture(autouse=True)
def _isolate_git_configuration(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")


def _git(repo, *arguments, check=True):
    return subprocess.run(
        ["git", *arguments], cwd=repo, check=check, capture_output=True, text=True
    )


def _commit(repo, content):
    (repo / "source.txt").write_text(content)
    _git(repo, "add", "source.txt")
    _git(
        repo,
        "-c", "user.name=Synthetic Test",
        "-c", "user.email=synthetic@example.invalid",
        "-c", "commit.gpgsign=false",
        "commit", "-m", "Independent synthetic fixture",
    )
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


@pytest.fixture
def upstream(tmp_path):
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main")
    older = _commit(repo, "Independent synthetic main source\n")
    _git(repo, "checkout", "-b", "pilot")
    selected = _commit(repo, "Independent synthetic pilot source\n")
    _git(repo, "checkout", "main")
    return repo, older, selected


def _source_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = {}
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, str(NOTEBOOK), "exec"), namespace)
    assert callable(namespace.get("checkout_source")), "Setup must expose pinned branch checkout"
    return namespace["checkout_source"]


def _assert_checkout(repo, commit, content):
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == commit
    assert _git(repo, "symbolic-ref", "-q", "HEAD", check=False).returncode == 1
    assert (repo / "source.txt").read_text() == content


def test_source_setup_pins_selected_branch_instead_of_remote_default(
    tmp_path, monkeypatch, upstream
):
    checkout = _source_checkout(tmp_path, monkeypatch)
    remote, _, selected = upstream
    repo, pin = tmp_path / "checkout", tmp_path / "source-pin.json"

    result = checkout(repo, str(remote), "pilot", "", pin)

    expected = {"repo_url": str(remote), "branch": "pilot", "commit": selected}
    assert result == expected
    assert json.loads(pin.read_text()) == expected
    _assert_checkout(repo, selected, "Independent synthetic pilot source\n")


@pytest.mark.parametrize("delete_checkout", [False, True], ids=["reuse", "reconstruct"])
def test_source_resume_preserves_pin_after_branch_advances(
    tmp_path, monkeypatch, upstream, delete_checkout
):
    checkout = _source_checkout(tmp_path, monkeypatch)
    remote, _, selected = upstream
    repo, pin = tmp_path / "checkout", tmp_path / "source-pin.json"
    original = checkout(repo, str(remote), "pilot", "", pin)
    saved = pin.read_bytes()
    _git(remote, "checkout", "pilot")
    advanced = _commit(remote, "Independent synthetic newer pilot source\n")
    assert advanced != selected
    if delete_checkout:
        shutil.rmtree(repo)

    assert checkout(repo, str(remote), "pilot", "", pin) == original

    assert pin.read_bytes() == saved
    _assert_checkout(repo, selected, "Independent synthetic pilot source\n")


def test_source_setup_accepts_exact_older_commit_override(tmp_path, monkeypatch, upstream):
    checkout = _source_checkout(tmp_path, monkeypatch)
    remote, older, _ = upstream
    repo, pin = tmp_path / "checkout", tmp_path / "source-pin.json"

    result = checkout(repo, str(remote), "pilot", older, pin)

    assert result == {"repo_url": str(remote), "branch": "pilot", "commit": older}
    assert json.loads(pin.read_text()) == result
    _assert_checkout(repo, older, "Independent synthetic main source\n")


@pytest.mark.parametrize("changed_field", ["repo_url", "branch", "commit"])
def test_source_resume_rejects_configuration_conflicting_with_pin(
    tmp_path, monkeypatch, upstream, changed_field
):
    checkout = _source_checkout(tmp_path, monkeypatch)
    remote, older, selected = upstream
    repo, pin = tmp_path / "checkout", tmp_path / "source-pin.json"
    checkout(repo, str(remote), "pilot", "", pin)
    saved = pin.read_bytes()
    arguments = {"repo_url": str(remote), "branch": "pilot", "code_ref": ""}
    if changed_field == "repo_url":
        arguments["repo_url"] = str(tmp_path / "different-source")
    elif changed_field == "branch":
        arguments["branch"] = "main"
    else:
        arguments["code_ref"] = older

    with pytest.raises(ValueError):
        checkout(repo, pin_path=pin, **arguments)

    assert pin.read_bytes() == saved
    _assert_checkout(repo, selected, "Independent synthetic pilot source\n")


@pytest.mark.parametrize("branch", ["", "../escape", "-pilot", "pilot:other"])
def test_source_setup_rejects_invalid_branch_before_writing_checkout(
    tmp_path, monkeypatch, upstream, branch
):
    checkout = _source_checkout(tmp_path, monkeypatch)
    remote, _, _ = upstream
    repo, pin = tmp_path / "checkout", tmp_path / "source-pin.json"

    with pytest.raises(ValueError):
        checkout(repo, str(remote), branch, "", pin)

    assert not repo.exists()
    assert not pin.exists()


@pytest.mark.parametrize("code_ref", ["pilot", "abcdef0", "g" * 40, "a" * 41])
def test_source_setup_rejects_non_exact_sha_before_writing_checkout(
    tmp_path, monkeypatch, upstream, code_ref
):
    checkout = _source_checkout(tmp_path, monkeypatch)
    remote, _, _ = upstream
    repo, pin = tmp_path / "checkout", tmp_path / "source-pin.json"

    with pytest.raises(ValueError):
        checkout(repo, str(remote), "pilot", code_ref, pin)

    assert not repo.exists()
    assert not pin.exists()


def test_source_resume_preserves_dirty_checkout_and_pin(tmp_path, monkeypatch, upstream):
    checkout = _source_checkout(tmp_path, monkeypatch)
    remote, _, selected = upstream
    repo, pin = tmp_path / "checkout", tmp_path / "source-pin.json"
    original = checkout(repo, str(remote), "pilot", "", pin)
    saved = pin.read_bytes()
    (repo / "source.txt").write_text("Independent uncommitted local edit\n")
    (repo / "untracked.txt").write_text("Independent untracked local work\n")
    remote.rename(tmp_path / "unavailable-upstream")

    assert checkout(repo, str(remote), "pilot", "", pin) == original

    assert pin.read_bytes() == saved
    _assert_checkout(repo, selected, "Independent uncommitted local edit\n")
    assert (repo / "untracked.txt").read_text() == "Independent untracked local work\n"


def test_source_resume_rejects_existing_checkout_at_another_commit(
    tmp_path, monkeypatch, upstream
):
    checkout = _source_checkout(tmp_path, monkeypatch)
    remote, older, _ = upstream
    repo, pin = tmp_path / "checkout", tmp_path / "source-pin.json"
    checkout(repo, str(remote), "pilot", "", pin)
    saved = pin.read_bytes()
    _git(repo, "checkout", "--detach", older)

    with pytest.raises(ValueError):
        checkout(repo, str(remote), "pilot", "", pin)

    assert pin.read_bytes() == saved
    _assert_checkout(repo, older, "Independent synthetic main source\n")


def test_source_setup_rejects_existing_checkout_without_durable_pin(
    tmp_path, monkeypatch, upstream
):
    checkout = _source_checkout(tmp_path, monkeypatch)
    remote, _, selected = upstream
    repo, pin = tmp_path / "checkout", tmp_path / "source-pin.json"
    _git(tmp_path, "clone", "--no-checkout", str(remote), str(repo))
    _git(repo, "checkout", "--detach", selected)
    (repo / "source.txt").write_text("Independent existing local work\n")

    with pytest.raises(ValueError):
        checkout(repo, str(remote), "pilot", "", pin)

    assert not pin.exists()
    _assert_checkout(repo, selected, "Independent existing local work\n")


@pytest.mark.parametrize("matching_metadata", [False, True])
def test_durable_git_binding_requires_the_checked_out_source_commit(
    tmp_path, monkeypatch, upstream, matching_metadata
):
    checkout = _source_checkout(tmp_path, monkeypatch)
    bind = checkout.__globals__.get("bind_git_metadata")
    assert callable(bind), "Durable Git restoration must validate the effective source commit"
    remote, older, selected = upstream
    repo, pin = tmp_path / "checkout", tmp_path / "source-pin.json"
    checkout(repo, str(remote), "pilot", "", pin)
    if matching_metadata:
        _git(remote, "checkout", "--detach", selected)
    durable = tmp_path / "durable-git"
    shutil.copytree(remote / ".git", durable)

    if matching_metadata:
        bind(repo, durable, selected)
        assert (repo / ".git").is_symlink()
        assert (repo / ".git").resolve() == durable.resolve()
    else:
        with pytest.raises(ValueError, match="commit|HEAD"):
            bind(repo, durable, selected)
        assert (repo / ".git").is_dir() and not (repo / ".git").is_symlink()
        assert _git(tmp_path, f"--git-dir={durable}", "rev-parse", "HEAD").stdout.strip() == older
    _assert_checkout(repo, selected, "Independent synthetic pilot source\n")
