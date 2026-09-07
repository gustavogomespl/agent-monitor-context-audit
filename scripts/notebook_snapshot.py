"""Checked public runtime snapshot embedded in the portable Colab notebook."""

SOURCE_PAYLOAD_B64 = globals().get("SOURCE_PAYLOAD_B64", "")
SOURCE_PAYLOAD_SHA256 = globals().get("SOURCE_PAYLOAD_SHA256", "")

def snapshot_path_allowed(name):
    from pathlib import PurePosixPath

    path = PurePosixPath(name)
    if (str(path) != name or path.is_absolute() or "\\" in name
            or any(part.startswith(".") or part == "private" for part in path.parts)):
        return False
    return (name in {"pyproject.toml", "uv.lock", "requirements-colab.txt"}
            or name in {"research_plan.md", "docs/decisions.md", "docs/qwen_colab.md"}
            or name in {"scripts/colab_bootstrap.py", "scripts/notebook_snapshot.py"}
            or (name.startswith("src/context_audit/") and path.suffix == ".py")
            or (name.startswith("prompts/") and path.suffix == ".txt"))


def scientific_source_hash(repo, replacements):
    import hashlib
    import json
    from pathlib import Path

    names = {str(p.relative_to(repo)) for p in (repo / "src").rglob("*.py")}
    names.update({"pyproject.toml", "uv.lock", "requirements-colab.txt"})
    names.update(name for name in replacements if name.startswith("src/"))
    content = {}
    for name in sorted(names):
        if name in replacements:
            content[name] = replacements[name].decode()
        elif (repo / Path(name)).exists():
            content[name] = (repo / name).read_text()
    return hashlib.sha256(json.dumps(
        content, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def apply_embedded_source(repo, drive_root, *, frozen=False, encoded=None, expected=None,
                          run_root=None, run_version=None):
    import base64
    import hashlib
    import json
    import os
    import tempfile
    import zlib

    if run_version is not None or run_root is not None:
        if (run_version not in {"summary-v2", "summary-v3"} or run_root is None
                or run_root.name != "runs-private"
                or drive_root.resolve() != run_root.parent.resolve() / "versions" / run_version):
            raise ValueError("A versioned source refresh requires its isolated version workspace.")
        manifests = [
            path for phase in ("pilot", "development", "test")
            for path in run_root.glob(f"qwen-{phase}-{run_version}*/manifests/run.json")
        ]
    else:
        manifests = (drive_root / "runs-private").glob("qwen-*/manifests/run.json")

    encoded = SOURCE_PAYLOAD_B64 if encoded is None else encoded
    expected = SOURCE_PAYLOAD_SHA256 if expected is None else expected
    raw = zlib.decompress(base64.b64decode(encoded, validate=True))
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Notebook source checksum mismatch; use an intact notebook.")
    payload = json.loads(raw)
    if payload["schema_version"] != 1 or not payload["files"]:
        raise ValueError("Invalid notebook source manifest.")
    repo = repo.resolve()
    old_receipt = drive_root / "configuration/notebook-source.json"
    old_hashes = (
        json.loads(old_receipt.read_text()).get("files", {}) if old_receipt.exists() else {}
    )
    pending, originals, contents = {}, {}, {}
    for item in payload["files"]:
        name = item["path"]
        if not snapshot_path_allowed(name) or name in contents:
            raise ValueError("Notebook source contains an invalid public path.")
        path = repo / name
        if any(p.is_symlink() for p in [path, *path.parents] if p != repo):
            raise ValueError("Source path is a symlink; preserve and review the local workspace.")
        body = item["text"].encode()
        if hashlib.sha256(body).hexdigest() != item["sha256"]:
            raise ValueError("Notebook source file checksum mismatch.")
        contents[name] = body
        previous = path.read_bytes() if path.exists() else None
        if previous == body:
            continue
        if frozen:
            raise ValueError(
                "Frozen source differs from this notebook; retain its original notebook."
            )
        accepted = [*item["accepted_previous_sha256"], old_hashes.get(name)]
        if previous is not None and hashlib.sha256(previous).hexdigest() not in accepted:
            raise ValueError(f"Preserving modified local source: {name}. Review before updating.")
        pending[name], originals[name] = body, previous

    # A source refresh cannot turn a recorded scientific run into different methods.
    resulting_hash = scientific_source_hash(repo, contents)
    for manifest in manifests:
        if json.loads(manifest.read_text())["code_hash"] != resulting_hash:
            raise ValueError("Recorded run code differs from this notebook. Preserve its results "
                             "and use a new explicitly exploratory workspace for changed methods.")

    def atomic(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".snapshot-")
        try:
            with os.fdopen(fd, "wb") as target:
                target.write(content)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    backup = drive_root / "configuration/source-backups" / expected
    written = []
    try:
        for name, body in pending.items():
            if originals[name] is not None:
                atomic(backup / name, originals[name])
            atomic(repo / name, body)
            written.append(name)
    except BaseException:
        for name in reversed(written):
            if originals[name] is None:
                (repo / name).unlink()
            else:
                atomic(repo / name, originals[name])
        raise
    receipt = dict(
        snapshot_sha256=expected, code_hash=resulting_hash, changed=sorted(pending),
        files={name: hashlib.sha256(body).hexdigest() for name, body in contents.items()},
    )
    atomic(drive_root / "configuration/notebook-source.json",
           json.dumps(receipt, indent=2).encode())
    print("Notebook runtime verified:", expected[:12], "| refreshed files:", len(pending))
    return receipt
