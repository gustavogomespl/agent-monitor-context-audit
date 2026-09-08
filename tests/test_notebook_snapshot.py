"""Portable notebooks carry checked public source, never private artifacts."""

import base64
import hashlib
import json
import runpy
import zlib
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/notebook_snapshot.py"


def payload(path="src/context_audit/example.py", original="old\n", updated="new\n"):
    body = {"schema_version": 1, "files": [{
        "path": path, "text": updated,
        "sha256": hashlib.sha256(updated.encode()).hexdigest(),
        "accepted_previous_sha256": [hashlib.sha256(original.encode()).hexdigest()],
    }]}
    raw = json.dumps(body).encode()
    return base64.b64encode(zlib.compress(raw)).decode(), hashlib.sha256(raw).hexdigest()


def apply(repo, drive, **kwargs):
    return runpy.run_path(str(SCRIPT))["apply_embedded_source"](repo, drive, **kwargs)


def test_snapshot_repairs_known_old_source_and_preserves_private_data(tmp_path):
    repo, drive = tmp_path / "repo", tmp_path / "drive"
    source = repo / "src/context_audit/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("old\n")
    private = repo / "data/private/sentinel"
    private.parent.mkdir(parents=True)
    private.write_text("Independent synthetic private content")
    encoded, expected = payload()
    apply(repo, drive, encoded=encoded, expected=expected)
    assert source.read_text() == "new\n"
    assert private.read_text() == "Independent synthetic private content"
    assert apply(repo, drive, encoded=encoded, expected=expected)["changed"] == []


@pytest.mark.parametrize("name", ["../escape.py", ".git/config", "data/private/file.py"])
def test_snapshot_rejects_nonpublic_paths_before_writing(tmp_path, name):
    encoded, expected = payload(name)
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(ValueError, match="public|path"):
        apply(repo, tmp_path / "drive", encoded=encoded, expected=expected)
    assert list(repo.iterdir()) == []


def test_snapshot_preserves_unknown_local_edits(tmp_path):
    source = tmp_path / "src/context_audit/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("Unrelated local edit")
    encoded, expected = payload()
    with pytest.raises(ValueError, match="local|modified"):
        apply(tmp_path, tmp_path / "drive", encoded=encoded, expected=expected)
    assert source.read_text() == "Unrelated local edit"


def test_frozen_snapshot_cannot_be_upgraded(tmp_path):
    source = tmp_path / "src/context_audit/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("old\n")
    encoded, expected = payload()
    with pytest.raises(ValueError, match="frozen|Frozen"):
        apply(tmp_path, tmp_path / "drive", encoded=encoded, expected=expected, frozen=True)
    assert source.read_text() == "old\n"


def test_snapshot_checks_payload_digest(tmp_path):
    encoded, _ = payload()
    with pytest.raises(ValueError, match="checksum"):
        apply(tmp_path, tmp_path / "drive", encoded=encoded, expected="0" * 64)


def test_recorded_scientific_code_cannot_be_changed_by_snapshot(tmp_path):
    source = tmp_path / "src/context_audit/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("old\n")
    drive = tmp_path / "drive"
    manifest = drive / "runs-private/qwen-pilot/manifests/run.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"code_hash": "0" * 64}))
    encoded, expected = payload()
    with pytest.raises(ValueError, match="Recorded run code"):
        apply(tmp_path, drive, encoded=encoded, expected=expected)
    assert source.read_text() == "old\n"


@pytest.mark.parametrize("version", [
    "summary-v2", "summary-v3", "summary-v4", "summary-v5", "summary-v6",
])
def test_explicit_version_uses_isolated_source_and_retains_legacy_run(tmp_path, version):
    repo, drive = tmp_path / "repo", tmp_path / "drive"
    source = repo / "src/context_audit/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("old\n")
    old = drive / "runs-private/qwen-pilot-ctx196608/manifests/run.json"
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({"code_hash": "old-run-hash"}))
    encoded, expected = payload()
    apply(repo, drive / "versions" / version, encoded=encoded, expected=expected,
          run_root=drive / "runs-private", run_version=version)
    assert source.read_text() == "new\n"
    assert json.loads(old.read_text()) == {"code_hash": "old-run-hash"}


@pytest.mark.parametrize("version", [
    "summary-v2", "summary-v3", "summary-v4", "summary-v5", "summary-v6",
])
def test_existing_version_run_still_blocks_scientific_source_drift(tmp_path, version):
    repo, drive = tmp_path / "repo", tmp_path / "drive"
    source = repo / "src/context_audit/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("old\n")
    active = drive / "runs-private" / f"qwen-pilot-{version}-ctx196608/manifests/run.json"
    active.parent.mkdir(parents=True)
    active.write_text(json.dumps({"code_hash": "prior-v2-hash"}))
    encoded, expected = payload()
    with pytest.raises(ValueError, match="Recorded run code"):
        apply(repo, drive / "versions" / version, encoded=encoded, expected=expected,
              run_root=drive / "runs-private", run_version=version)
    assert source.read_text() == "old\n"


@pytest.mark.parametrize("version", [
    "summary-v2", "summary-v3", "summary-v4", "summary-v5", "summary-v6",
])
def test_version_cannot_bypass_source_guard_in_legacy_workspace(tmp_path, version):
    encoded, expected = payload()
    tmp_path.joinpath("repo").mkdir()
    with pytest.raises(ValueError, match="isolated|version"):
        apply(tmp_path / "repo", tmp_path / "drive", encoded=encoded, expected=expected,
              run_root=tmp_path / "drive/runs-private", run_version=version)


def test_v3_snapshot_preserves_recorded_v2_source_and_results(tmp_path):
    repo, drive = tmp_path / "repo", tmp_path / "drive"
    source = repo / "src/context_audit/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("old\n")
    v2 = drive / "runs-private/qwen-pilot-summary-v2-ctx196608/manifests/run.json"
    v2.parent.mkdir(parents=True)
    v2.write_text(json.dumps({"code_hash": "synthetic-v2-hash"}))
    prior_source = drive / "versions/summary-v2/configuration/notebook-source.json"
    prior_source.parent.mkdir(parents=True)
    prior_source.write_text(json.dumps({"snapshot_sha256": "synthetic-v2-snapshot"}))
    old = {v2: v2.read_bytes(), prior_source: prior_source.read_bytes()}
    encoded, expected = payload()
    apply(repo, drive / "versions/summary-v3", encoded=encoded, expected=expected,
          run_root=drive / "runs-private", run_version="summary-v3")
    assert source.read_text() == "new\n"
    for path, content in old.items():
        assert path.read_bytes() == content


def test_v4_snapshot_preserves_all_previous_source_receipts_and_results(tmp_path):
    repo, drive = tmp_path / "repo", tmp_path / "drive"
    source = repo / "src/context_audit/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("old\n")
    old = {}
    for version in ("summary-v2", "summary-v3"):
        manifest = drive / f"runs-private/qwen-pilot-{version}-ctx196608/manifests/run.json"
        receipt = drive / f"versions/{version}/configuration/notebook-source.json"
        for path, content in (
            (manifest, {"code_hash": f"synthetic-{version}-hash"}),
            (receipt, {"snapshot_sha256": f"synthetic-{version}-snapshot"}),
        ):
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(content))
            old[path] = path.read_bytes()
    encoded, expected = payload()
    result = apply(repo, drive / "versions/summary-v4", encoded=encoded, expected=expected,
                   run_root=drive / "runs-private", run_version="summary-v4")
    assert source.read_text() == "new\n"
    assert result["snapshot_sha256"] == expected
    receipt = drive / "versions/summary-v4/configuration/notebook-source.json"
    assert json.loads(receipt.read_text())["snapshot_sha256"] == expected
    for path, content in old.items():
        assert path.read_bytes() == content


def test_v5_snapshot_preserves_all_previous_source_receipts_and_results(tmp_path):
    repo, drive = tmp_path / "repo", tmp_path / "drive"
    source = repo / "src/context_audit/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("old\n")
    old = {}
    for version in ("summary-v2", "summary-v3", "summary-v4"):
        manifest = drive / f"runs-private/qwen-pilot-{version}-ctx196608/manifests/run.json"
        receipt = drive / f"versions/{version}/configuration/notebook-source.json"
        for path, content in (
            (manifest, {"code_hash": f"synthetic-{version}-hash"}),
            (receipt, {"snapshot_sha256": f"synthetic-{version}-snapshot"}),
        ):
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(content))
            old[path] = path.read_bytes()
    encoded, expected = payload()
    result = apply(repo, drive / "versions/summary-v5", encoded=encoded, expected=expected,
                   run_root=drive / "runs-private", run_version="summary-v5")
    assert source.read_text() == "new\n"
    assert result["snapshot_sha256"] == expected
    receipt = drive / "versions/summary-v5/configuration/notebook-source.json"
    assert json.loads(receipt.read_text())["snapshot_sha256"] == expected
    for path, content in old.items():
        assert path.read_bytes() == content


def test_v6_snapshot_preserves_all_previous_source_receipts_and_results(tmp_path):
    repo, drive = tmp_path / "repo", tmp_path / "drive"
    source = repo / "src/context_audit/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("old\n")
    old = {}
    for version in ("summary-v2", "summary-v3", "summary-v4", "summary-v5"):
        manifest = drive / f"runs-private/qwen-pilot-{version}-ctx196608/manifests/run.json"
        receipt = drive / f"versions/{version}/configuration/notebook-source.json"
        for path, content in (
            (manifest, {"code_hash": f"synthetic-{version}-hash"}),
            (receipt, {"snapshot_sha256": f"synthetic-{version}-snapshot"}),
        ):
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(content))
            old[path] = path.read_bytes()
    encoded, expected = payload()
    result = apply(repo, drive / "versions/summary-v6", encoded=encoded, expected=expected,
                   run_root=drive / "runs-private", run_version="summary-v6")
    assert source.read_text() == "new\n"
    assert result["snapshot_sha256"] == expected
    receipt = drive / "versions/summary-v6/configuration/notebook-source.json"
    assert json.loads(receipt.read_text())["snapshot_sha256"] == expected
    for path, content in old.items():
        assert path.read_bytes() == content


@pytest.mark.parametrize("version", [
    "summary-v2", "summary-v3", "summary-v4", "summary-v5", "summary-v6",
])
def test_version_source_guard_requires_the_actual_shared_runs_directory(tmp_path, version):
    repo, drive = tmp_path / "repo", tmp_path / "drive"
    repo.mkdir()
    encoded, expected = payload()
    with pytest.raises(ValueError, match="isolated|version"):
        apply(repo, drive / "versions" / version, encoded=encoded, expected=expected,
              run_root=drive / "unrelated-empty-directory", run_version=version)
    assert list(repo.iterdir()) == []


def test_source_symlink_cannot_escape_into_private_storage(tmp_path):
    target = tmp_path / "private"
    target.mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "src/context_audit").symlink_to(target, target_is_directory=True)
    encoded, expected = payload()
    with pytest.raises(ValueError, match="symlink"):
        apply(tmp_path, tmp_path / "drive", encoded=encoded, expected=expected)
    assert list(target.iterdir()) == []


def test_delivered_notebook_contains_current_public_source_only():
    import nbformat

    root = SCRIPT.parents[1]
    notebook = nbformat.read(root / "notebooks/03_qwen_colab.ipynb", as_version=4)
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(cell.source, namespace)
    raw = zlib.decompress(base64.b64decode(namespace["SOURCE_PAYLOAD_B64"]))
    assert hashlib.sha256(raw).hexdigest() == namespace["SOURCE_PAYLOAD_SHA256"]
    paths = set()
    for record in json.loads(raw)["files"]:
        name = record["path"]
        assert namespace["snapshot_path_allowed"](name)
        assert record["text"] == (root / name).read_text()
        assert hashlib.sha256(record["text"].encode()).hexdigest() == record["sha256"]
        paths.add(name)
    assert {"src/context_audit/colab.py", "scripts/colab_bootstrap.py", "uv.lock"} <= paths
    # The eventual Colab freeze must include the amended methods and decision record.
    assert {"research_plan.md", "docs/decisions.md", "docs/qwen_colab.md"} <= paths
