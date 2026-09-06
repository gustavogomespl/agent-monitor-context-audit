"""Atomic private JSON cache and fsynced accounting journals (single process)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


class PrivateStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def path(self, namespace: str, key: str | None = None) -> Path:
        for part in (namespace, key):
            if part is not None and not re.fullmatch(r"[a-zA-Z0-9_-]+", part):
                raise ValueError("Invalid private store key")
        if key is None:
            return self.root / f"{namespace}.jsonl"
        directory = self.root / namespace
        directory.mkdir(exist_ok=True, mode=0o700)
        return directory / f"{key}.json"

    def get(self, namespace: str, key: str) -> dict | None:
        path = self.path(namespace, key)
        return json.loads(path.read_text()) if path.exists() else None

    def put(self, namespace: str, key: str, value: dict) -> None:
        path = self.path(namespace, key)
        fd, name = tempfile.mkstemp(dir=path.parent, prefix=".atomic-")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(value, handle, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def append(self, namespace: str, value: dict) -> None:
        path = self.path(namespace)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def read_journal(self, namespace: str) -> list[dict]:
        path = self.path(namespace)
        if not path.exists():
            return []
        try:
            return [json.loads(line) for line in path.read_text().splitlines()]
        except (ValueError, TypeError) as exc:
            raise ValueError("Corrupt private journal; reconcile before resuming") from exc
