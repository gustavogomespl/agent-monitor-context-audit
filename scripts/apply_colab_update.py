"""Apply a reviewed public-source update without changing the pinned Git HEAD.

Uses only the Python standard library and Git, so it can run before reloading the
installed package. The archive is read directly; arbitrary ZIP paths are never
extracted. Every target and payload is validated before source files are changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def source_path_allowed(value: str) -> bool:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or str(path) != value
        or any(part.startswith(".") or part == "private" for part in path.parts)
        or "\\" in value
    ):
        return False
    if value in {"README.md", "research_plan.md", "pyproject.toml", "uv.lock"}:
        return True
    if value == "notebooks/03_qwen_colab.ipynb":
        return True
    return (
        (value.startswith("src/context_audit/") and path.suffix == ".py")
        or (path.parts[0] in {"tests", "scripts"} and path.suffix == ".py")
        or (path.parts[0] == "docs" and path.suffix == ".md")
    )


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _atomic_write(path: Path, content: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def apply_update(repo: Path, archive: Path) -> dict:
    repo, archive = repo.resolve(), archive.resolve()
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    pending = []
    observed = []
    with zipfile.ZipFile(archive) as source:
        if len(source.namelist()) != len(set(source.namelist())):
            raise ValueError("Duplicate archive paths are not allowed")
        manifest = json.loads(source.read("manifest.json"))
        if manifest.get("schema_version") != 1 or not re.fullmatch(
            r"[0-9a-f]{40}", manifest.get("base_commit", "")
        ):
            raise ValueError("Invalid source update manifest")
        if head != manifest["base_commit"]:
            raise ValueError("Git HEAD differs from the update's pinned base commit")
        records = manifest.get("files")
        if not isinstance(records, list) or not records:
            raise ValueError("Update manifest has no public source files")
        seen = set()
        for record in records:
            name = record.get("path", "")
            if not isinstance(name, str) or not source_path_allowed(name) or name in seen:
                raise ValueError(f"Invalid or repeated public source path: {name!r}")
            seen.add(name)
            path = repo / name
            if any(parent.is_symlink() for parent in [path, *path.parents] if parent != repo):
                raise ValueError(f"Source symlink path refused: {name}")
            if not path.resolve().is_relative_to(repo):
                raise ValueError(f"Source path escapes repository: {name}")
            body = source.read(f"payload/{name}")
            desired = record.get("sha256")
            if not isinstance(desired, str) or sha256(body) != desired:
                raise ValueError(f"Update payload hash mismatch: {name}")
            current = path.read_bytes() if path.exists() else None
            before = sha256(current) if current is not None else None
            accepted = record.get("accepted_sha256")
            if not isinstance(accepted, list) or before not in [desired, *accepted]:
                raise ValueError(f"Locally modified source hash refused: {name}")
            mode = path.stat().st_mode & 0o777 if current is not None else 0o644
            observed.append({"path": name, "before_sha256": before, "after_sha256": desired})
            if before != desired:
                pending.append((path, current, body, mode))
    # Keep the update receipt in the existing gitignored private run workspace.
    receipt_root = repo / "runs/private/source-updates"
    ignored = subprocess.run(
        # Colab binds runs/private itself to Drive; Git rejects a path beyond that symlink.
        ["git", "-C", str(repo), "check-ignore", "--quiet", "runs/private"]
    )
    if ignored.returncode:
        raise ValueError("Source update receipt directory must already be gitignored")
    changed = []
    try:
        for path, previous, body, mode in pending:
            _atomic_write(path, body, mode)
            changed.append((path, previous, mode))
    except BaseException:
        for path, previous, mode in reversed(changed):
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                _atomic_write(path, previous, mode)
        raise
    digest = sha256(archive.read_bytes())
    receipt = receipt_root / f"{digest}.json"
    result = {
        "schema_version": 1,
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "base_commit": head,
        "archive_sha256": digest,
        "updated_files": [path.relative_to(repo).as_posix() for path, _, _, _ in pending],
        "files": observed,
        "receipt": str(receipt),
        "git_head_preserved": True,
    }
    if not receipt.exists():
        _atomic_write(receipt, (json.dumps(result, indent=2) + "\n").encode())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(apply_update(args.repo, args.archive), indent=2))


if __name__ == "__main__":
    main()
