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
