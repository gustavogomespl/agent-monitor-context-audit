"""Source verification against independent synthetic local Git repositories."""

import subprocess

import pytest

from context_audit import dataset


def _git(repo, *arguments, check=True):
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def source_repository(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    repo = tmp_path / "independent-synthetic-source"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "core.fileMode", "true")
    _git(repo, "config", "user.name", "Synthetic Test")
    _git(repo, "config", "user.email", "synthetic@example.invalid")
    _git(repo, "config", "commit.gpgsign", "false")
    tracked = repo / "source.txt"
    tracked.write_text("Independent synthetic original content\n")
    tracked.chmod(0o644)
    _git(repo, "add", "source.txt")
    _git(repo, "commit", "--quiet", "-m", "Independent synthetic fixture")
    commit = _git(repo, "rev-parse", "HEAD").stdout.strip()
    remote = str(tmp_path / "independent-synthetic-remote.git")
    _git(repo, "remote", "add", "origin", remote)
    monkeypatch.setattr(dataset, "DATASET_COMMIT", commit)
    monkeypatch.setattr(dataset, "DATASET_REPOSITORY", remote)
    return repo, commit


def test_source_verification_accepts_only_worktree_executable_bit_drift(source_repository):
    repo, commit = source_repository
    assert dataset._verified_source_commit(repo) == commit
    (repo / "source.txt").chmod(0o755)

    assert dataset._verified_source_commit(repo) == commit
    assert _git(repo, "config", "--get", "core.fileMode").stdout.strip() == "true"


@pytest.mark.parametrize("mutation", ["content", "deletion", "staged_content", "staged_mode"])
def test_source_verification_rejects_tracked_changes_despite_mode_drift(
    source_repository, mutation
):
    repo, _ = source_repository
    tracked = repo / "source.txt"
    tracked.chmod(0o755)
    if mutation in {"content", "staged_content"}:
        tracked.write_text("Independent synthetic changed content, longer than before\n")
    elif mutation == "deletion":
        tracked.unlink()
    if mutation == "staged_content":
        _git(repo, "add", "--chmod=-x", "source.txt")
    elif mutation == "staged_mode":
        _git(repo, "add", "source.txt")

    with pytest.raises(dataset.DatasetError, match="Tracked official source files changed"):
        dataset._verified_source_commit(repo)


def test_source_verification_rejects_changed_commit(source_repository):
    repo, _ = source_repository
    _git(repo, "commit", "--quiet", "--allow-empty", "-m", "Another synthetic commit")

    with pytest.raises(dataset.DatasetError, match="Source commit differs"):
        dataset._verified_source_commit(repo)


def test_source_verification_rejects_changed_remote(source_repository, tmp_path):
    repo, _ = source_repository
    _git(repo, "remote", "set-url", "origin", str(tmp_path / "other-synthetic-source.git"))

    with pytest.raises(dataset.DatasetError, match="Source remote is not"):
        dataset._verified_source_commit(repo)
