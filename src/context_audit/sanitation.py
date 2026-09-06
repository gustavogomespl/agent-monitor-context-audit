"""Pre-publication text scan, including decoded notebook strings and outputs.

This is a defense in depth content check, not proof against paraphrase or image leaks.
Only strict numeric exports and independently written fixtures should be published.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _texts(path: Path):
    try:
        raw = path.read_text()
    except (UnicodeError, OSError):
        return
    yield raw
    try:
        yield from _strings(json.loads(raw))
    except ValueError:
        if path.suffix == ".jsonl":
            for line in raw.splitlines():
                try:
                    yield from _strings(json.loads(line))
                except ValueError:
                    continue


def _normalize(text):
    return re.sub(r"\s+", " ", text)


def scan_public_files(paths: list[Path], protected_paths: list[Path], *, root: Path) -> list[dict]:
    # Hash windows so scan evidence never republishes source text.
    protected = set()
    canary_identifiers = set()
    for path in protected_paths:
        for text in _texts(path):
            value = _normalize(text)
            canary_identifiers.update(re.findall(r"canary GUID ([A-Za-z0-9-]{16,})", text))
            for start in range(0, len(value) - 79, 40):
                protected.add(hashlib.sha256(value[start : start + 80].encode()).digest())
    findings = []
    for path in paths:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        if (
            relative.startswith(("data/private/", "runs/private/", "notebooks/private/"))
            or path.name == ".env"
            or path.name.endswith(".local.yaml")
        ):
            findings.append(dict(path=relative, reason="private_path"))
            continue
        if path.suffix == ".ipynb":
            try:
                notebook = json.loads(path.read_text())
                if any(cell.get("outputs") for cell in notebook.get("cells", [])):
                    findings.append(dict(path=relative, reason="notebook_outputs_present"))
            except ValueError:
                findings.append(dict(path=relative, reason="invalid_notebook"))
        for text in _texts(path):
            if any(identifier in text for identifier in canary_identifiers):
                findings.append(dict(path=relative, reason="source_canary_match"))
            if re.search(r"\bsk-ant-[A-Za-z0-9_-]{16,}", text):
                findings.append(dict(path=relative, reason="possible_api_secret"))
            value = _normalize(text)
            if protected and any(
                hashlib.sha256(value[i : i + 80].encode()).digest() in protected
                for i in range(len(value) - 79)
            ):
                findings.append(dict(path=relative, reason="protected_content_match"))
                break
    return findings


def scan_repository(root: Path = Path(".")) -> dict:
    output = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout
    paths = [root / p.decode() for p in output.split(b"\0") if p]
    paths = [p for p in paths if p.is_file()]
    protected = [
        *root.glob("data/private/upstream/attacks/**/*.jsonl"),
        *root.glob("data/private/upstream/attacks/**/metadata.json"),
        *root.glob("data/private/upstream/attacks/**/description.md"),
        *root.glob("runs/private/**/calls/*.json"),
        *root.glob("runs/private/**/representations/*.json"),
    ]
    # Inventory/count manifests are intentionally public aggregates, not source text.
    # Request system prompts are public templates; source histories are scanned above.
    findings = scan_public_files(paths, protected, root=root)
    return dict(
        status="passed" if not findings else "failed",
        public_files=len(paths),
        protected_files=len(protected),
        protected_corpus_available=bool(protected),
        findings=findings,
        limitations="Exact text windows only; paraphrases and raster figures need review",
    )
