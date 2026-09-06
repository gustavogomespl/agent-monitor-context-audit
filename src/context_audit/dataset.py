"""Read official JSONL as data and keep all benchmark-derived content private."""

from __future__ import annotations

import hashlib
import hmac
import json
import random
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from context_audit.schemas import DataOrigin, EvaluationLabel, Event, TranscriptInput


class DatasetError(ValueError):
    """Malformed or incomplete data; never replaced silently with fixtures."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"Invalid JSON at line {number}") from exc
        if not isinstance(record, dict):
            raise DatasetError(f"Expected JSON object at line {number}")
        records.append(record)
    return records


def extract_canaries(path: Path) -> list[str]:
    """Read the official canary objects separately from model-visible inputs."""
    return [
        row["canary"]
        for row in _read_jsonl(path)
        if "canary" in row and "role" not in row and isinstance(row["canary"], str)
    ]


def _without_thinking(value: Any) -> Any:
    if isinstance(value, list):
        return [
            _without_thinking(x)
            for x in value
            if not isinstance(x, dict) or x.get("type") not in ("thinking", "redacted_thinking")
        ]
    if isinstance(value, dict):
        if value.get("type") in ("thinking", "redacted_thinking"):
            return None
        return {k: _without_thinking(v) for k, v in value.items()}
    return value


def parse_transcript(
    path: Path, transcript_id: str, data_origin: DataOrigin = "sleight_bench"
) -> TranscriptInput:
    """Normalize only visible blocks; never read evaluator sidecars or directory names.

    Raises on malformed visible events, missing initial human text, duplicate or orphan
    tool IDs. The initial task is removed once, before assigning neutral event IDs.
    """
    records = _read_jsonl(path)
    if data_origin == "sleight_bench" and (
        not records or "canary" not in records[0] or "role" in records[0]
    ):
        raise DatasetError("Real transcripts require their official leading canary")
    initial_task = None
    initial_cwd = None
    events = []
    calls: set[str] = set()
    for number, row in enumerate(records, 1):
        if "canary" in row and "role" not in row:
            continue
        role, cwd, content = row.get("role"), row.get("cwd"), row.get("content")
        if role not in ("user", "assistant") or not isinstance(content, dict):
            raise DatasetError(f"Invalid single-block message at line {number}")
        if not isinstance(cwd, str) or not cwd.strip():
            raise DatasetError(f"Missing cwd at line {number}")
        kind = content.get("type")
        if kind in ("thinking", "redacted_thinking"):
            continue
        if initial_task is None:
            if role != "user" or kind != "text" or not isinstance(content.get("text"), str):
                raise DatasetError("First visible event must be real user text, not a tool result")
            initial_task = content["text"]
            initial_cwd = cwd
            if not initial_task.strip():
                raise DatasetError("Initial task is empty")
            continue
        values: dict[str, Any] = {
            "event_id": f"E{len(events) + 1:04d}",
            "role": role,
            "kind": kind,
            "cwd": cwd,
        }
        if kind == "text":
            values["text"] = content.get("text")
        elif kind == "tool_use":
            call_id = content.get("id")
            if not isinstance(call_id, str) or call_id in calls:
                raise DatasetError(f"Invalid or duplicate tool ID at line {number}")
            calls.add(call_id)
            values.update(
                tool_name=content.get("name"),
                tool_use_id=call_id,
                arguments=_without_thinking(content.get("input")),
            )
        elif kind == "tool_result":
            call_id = content.get("tool_use_id")
            if call_id not in calls:
                raise DatasetError(f"Orphan tool result at line {number}")
            values.update(
                tool_use_id=call_id,
                result=_without_thinking(content.get("content")),
                is_error=content.get("is_error"),
            )
        else:
            raise DatasetError(f"Unsupported visible block at line {number}")
        try:
            events.append(Event.model_validate(values))
        except ValidationError as exc:
            raise DatasetError(f"Invalid visible block at line {number}") from exc
    if initial_task is None:
        raise DatasetError("No initial human task in transcript")
    return TranscriptInput(
        transcript_id=transcript_id,
        initial_task=initial_task,
        initial_cwd=initial_cwd,
        events=events,
        data_origin=data_origin,
    )


def opaque_id(secret: bytes, namespace: str, source_identity: str) -> str:
    if len(secret) < 32:
        raise ValueError("Opaque identifiers require a private secret of at least 32 bytes")
    prefixes = {"transcript": "t", "scenario": "s", "family": "f"}
    digest = hmac.new(secret, (namespace + "\0" + source_identity).encode(), hashlib.sha256)
    return prefixes[namespace] + "_" + digest.hexdigest()[:24]


def validate_splits(labels: list[EvaluationLabel], *, allow_unassigned: bool = False) -> None:
    ids = [row.transcript_id for row in labels]
    if len(ids) != len(set(ids)):
        raise DatasetError("Duplicate transcript IDs in evaluator table")
    family_splits: dict[str, set[str]] = {}
    scenarios: dict[str, list[EvaluationLabel]] = {}
    for row in labels:
        if row.split == "unassigned" and not allow_unassigned:
            raise DatasetError("Unassigned split")
        family_splits.setdefault(row.family_id, set()).add(row.split)
        scenarios.setdefault(row.scenario_id, []).append(row)
    if any(len(splits) != 1 for splits in family_splits.values()):
        raise DatasetError("Family overlap across splits")
    for members in scenarios.values():
        if sorted(row.label for row in members) != [0, 1]:
            raise DatasetError("Each scenario must have exactly one complete binary pair")
        if len({row.family_id for row in members}) != 1:
            raise DatasetError("Scenario members must share a family")


def assign_splits(
    labels: list[EvaluationLabel], *, seed: int, development_pairs: int = 8
) -> list[EvaluationLabel]:
    validate_splits(labels, allow_unassigned=True)
    if development_pairs < 1:
        raise DatasetError("Development must contain at least one pair")
    families: dict[str, list[EvaluationLabel]] = {}
    for row in labels:
        families.setdefault(row.family_id, []).append(row)
    ordered = sorted(families)
    random.Random(seed).shuffle(ordered)
    if len(ordered) < 2:
        raise DatasetError("Need at least two independent families for development and test")
    # Choose the closest achievable count while keeping each family intact and test nonempty.
    choices: dict[int, tuple[str, ...]] = {0: ()}
    for family in ordered:
        size = len(families[family]) // 2
        for count, selected in list(choices.items()):
            choices.setdefault(count + size, (*selected, family))
    total = len(labels) // 2
    candidates = [count for count in choices if 0 < count < total]
    best = min(candidates, key=lambda count: (abs(count - development_pairs), count))
    development = set(choices[best])
    assigned = [
        row.model_copy(update={"split": "development" if row.family_id in development else "test"})
        for row in sorted(labels, key=lambda row: row.transcript_id)
    ]
    validate_splits(assigned)
    return assigned


DATASET_REPOSITORY = "https://github.com/safety-research/sleight-bench.git"
DATASET_COMMIT = "218c58315cc01ff0dc5a100e906c27d82d259521"


def _private_dir(path: Path) -> Path:
    resolved = Path(path).resolve()
    if "private" not in resolved.parts[2:]:
        raise DatasetError(
            "Benchmark artifacts require a directory under a named private directory"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _verified_source_commit(upstream: Path) -> str:
    import subprocess

    try:
        actual = subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DatasetError("Expected the pinned official source checkout") from exc
    if actual != DATASET_COMMIT:
        raise DatasetError("Source commit differs from the reviewed dataset pin")
    remote = subprocess.check_output(
        ["git", "-C", str(upstream), "remote", "get-url", "origin"], text=True
    ).strip()
    if remote.removesuffix(".git") != DATASET_REPOSITORY.removesuffix(".git"):
        raise DatasetError("Source remote is not the official benchmark repository")
    pristine = subprocess.run(
        ["git", "-C", str(upstream), "diff", "--quiet", "HEAD", "--"], check=False
    )
    if pristine.returncode:
        raise DatasetError("Tracked official source files changed after acquisition")
    return actual


def _decrypt_payload(path: Path, key: str) -> dict[str, bytes]:
    import base64

    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:
        raise DatasetError("Real acquisition and inventory require uv sync --extra data") from exc
    payload = json.loads(Fernet(key.encode()).decrypt(path.read_bytes()))
    return {name: base64.b64decode(value) for name, value in payload.items()}


def _verify_decrypted_sources(upstream: Path) -> None:
    """Reject edited or added source content, including gitignored decrypted files."""
    import re

    match = re.search(r"Decryption key: `([^`]+)`", (upstream / "README.md").read_text())
    if match is None:
        raise DatasetError("Official README has no documented decryption key")
    expected: set[Path] = set()
    encrypted_files = list((upstream / "attacks").rglob("encrypted.bin"))
    if not encrypted_files:
        raise DatasetError("Pinned encrypted source files are missing")
    for encrypted in encrypted_files:
        for name, content in _decrypt_payload(encrypted, match.group(1)).items():
            if Path(name).name != name or name in (".", ".."):
                raise DatasetError("Unsafe filename in encrypted source payload")
            path = encrypted.parent / name
            expected.add(path)
            if not path.is_file() or path.read_bytes() != content:
                raise DatasetError("Decrypted source differs from pinned encrypted payload")
    actual = set((upstream / "attacks").rglob("*.jsonl"))
    actual.update((upstream / "attacks").rglob("metadata.json"))
    actual.update((upstream / "attacks").rglob("description.md"))
    if actual - expected:
        raise DatasetError("Unverified extra source content is absent from encrypted payloads")


def _write_private_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    path.chmod(0o600)


def inventory_dataset(
    upstream: Path,
    private_dir: Path,
    public_dir: Path,
    *,
    seed: int = 20260905,
    development_pairs: int = 8,
) -> dict[str, Any]:
    """Inventory a reviewed, decrypted checkout; publish aggregate and opaque data only.

    Family links are the transitive closure of stage-name relationships, identical
    initial tasks/scenario annotations, and explicit sidecar references. Sidecars
    are used by this evaluator-only inventory, never by the transcript parser.
    """
    import csv
    import re
    import secrets
    from collections import Counter

    from context_audit.render import render_body

    upstream = Path(upstream).resolve()
    private_dir = _private_dir(private_dir)
    public_dir = Path(public_dir)
    commit = _verified_source_commit(upstream)
    _verify_decrypted_sources(upstream)
    attacks = upstream / "attacks"
    directories = sorted(path.parent for path in attacks.rglob("metadata.json"))
    if not directories or not any(attacks.rglob("transcript.jsonl")):
        raise DatasetError("No decrypted benchmark data; follow the official acquisition procedure")
    key_path = private_dir / "opaque_id_key.bin"
    if not key_path.exists():
        key_path.write_bytes(secrets.token_bytes(32))
        key_path.chmod(0o600)
    secret = key_path.read_bytes()
    canaries: set[str] = set()
    root_canary = upstream / "CANARY.txt"
    if root_canary.exists():
        canaries.add(root_canary.read_text())
    members: list[dict[str, Any]] = []
    normalized: dict[str, TranscriptInput] = {}
    parent = list(range(len(directories)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        left, right = find(a), find(b)
        if left != right:
            parent[max(left, right)] = min(left, right)

    stage_pattern = r"[-_](?:stage|part|session)[-_ ]?[0-9]+$"
    multi_pattern = r"multi[-_ ](?:stage|session)|(?:stage|part|session)[-_ ]*[12]"
    relation_keys: dict[tuple[str, str], int] = {}
    for index, directory in enumerate(directories):
        rel = directory.relative_to(attacks).as_posix()
        metadata = json.loads((directory / "metadata.json").read_text())
        if not isinstance(metadata, dict):
            raise DatasetError("Metadata is not an evaluator object")
        if isinstance(metadata.get("canary"), str):
            canaries.add(metadata["canary"])
        description = directory / "description.md"
        annotation_text = json.dumps(metadata) + (
            description.read_text() if description.exists() else ""
        )
        stem = re.sub(stage_pattern, "", rel, flags=re.IGNORECASE)
        structural_stage = stem != rel
        marked_multi = bool(re.search(multi_pattern, annotation_text, re.IGNORECASE))
        pair: list[dict[str, Any]] = []
        errors: list[str] = []
        for filename, label in [("transcript.jsonl", 1), ("benign.jsonl", 0)]:
            source = directory / filename
            if not source.exists():
                continue
            transcript_id = opaque_id(secret, "transcript", rel + "/" + filename)
            canaries.update(extract_canaries(source))
            info = {
                "transcript_id": transcript_id,
                "label": label,
                "source": str(source.relative_to(upstream)),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
            try:
                parsed = parse_transcript(source, transcript_id)
                normalized[transcript_id] = parsed
                info["body_utf8_bytes"] = len(render_body(parsed).encode())
                key = ("initial_task", re.sub(r"\s+", " ", parsed.initial_task).strip())
                if key in relation_keys:
                    union(index, relation_keys[key])
                relation_keys.setdefault(key, index)
            except DatasetError as exc:
                errors.append(str(exc))
            pair.append(info)
        for kind, value in [("stage_stem", stem), ("scenario", metadata.get("scenario"))]:
            if isinstance(value, str) and value.strip():
                key = (kind, re.sub(r"\s+", " ", value).strip())
                if key in relation_keys:
                    union(index, relation_keys[key])
                relation_keys.setdefault(key, index)
        members.append(
            {
                "source_directory": rel,
                "scenario_id": opaque_id(secret, "scenario", rel),
                "stage_stem": stem,
                "multisession_signal": structural_stage or marked_multi,
                "pair": pair,
                "errors": errors,
                "annotation_text": annotation_text,
            }
        )
    explicit_links = 0
    for i, member in enumerate(members):
        for j, directory in enumerate(directories):
            if i != j and directory.name in member["annotation_text"]:
                union(i, j)
                explicit_links += 1
    family_keys: dict[int, str] = {}
    for index, member in enumerate(members):
        family_keys.setdefault(find(index), member["source_directory"])
    multi_families = {find(i) for i, row in enumerate(members) if row["multisession_signal"]}
    labels: list[EvaluationLabel] = []
    exclusions = []
    mapping = []
    reason_counts: Counter[str] = Counter()
    for index, member in enumerate(members):
        family_id = opaque_id(secret, "family", family_keys[find(index)])
        if find(index) in multi_families:
            reason = "multisession_or_related"
        elif len(member["pair"]) != 2:
            reason = "missing_paired_transcript"
        elif member["errors"]:
            reason = "invalid_official_format"
        else:
            reason = None
        if reason:
            reason_counts[reason] += 1
            exclusions.append(
                {
                    "scenario_id": member["scenario_id"],
                    "reason": reason,
                    "source_transcript_count": len(member["pair"]),
                }
            )
        else:
            labels.extend(
                EvaluationLabel(
                    transcript_id=row["transcript_id"],
                    label=row["label"],
                    scenario_id=member["scenario_id"],
                    family_id=family_id,
                    split="unassigned",
                )
                for row in member["pair"]
            )
        mapping.append(
            {key: value for key, value in member.items() if key != "annotation_text"}
            | {"family_id": family_id, "exclusion_reason": reason}
        )
    assigned = assign_splits(labels, seed=seed, development_pairs=development_pairs)
    normalized_dir = private_dir / "normalized"
    normalized_dir.mkdir(exist_ok=True)
    normalized_hashes: dict[str, str] = {}
    input_hashes: dict[str, str] = {}
    canary_values = sorted(canaries)
    if not canary_values:
        raise DatasetError("No source canaries available for private derived artifacts")
    for label in assigned:
        transcript = normalized[label.transcript_id]
        _write_private_json(
            normalized_dir / (label.transcript_id + ".json"),
            {"canaries": canary_values, "input": transcript.model_dump(mode="json")},
        )
        normalized_hashes[label.transcript_id] = hashlib.sha256(
            (normalized_dir / (label.transcript_id + ".json")).read_bytes()
        ).hexdigest()
        input_hashes[label.transcript_id] = hashlib.sha256(
            transcript.model_dump_json().encode()
        ).hexdigest()
    label_path = private_dir / "labels.jsonl"
    label_path.write_text(
        json.dumps({"canaries": canary_values})
        + "\n"
        + "".join(json.dumps(row.model_dump()) + "\n" for row in assigned)
    )
    label_path.chmod(0o600)
    _write_private_json(private_dir / "canaries.json", {"canaries": canary_values})
    _write_private_json(
        private_dir / "source_mapping.json",
        {"canaries": canary_values, "dataset_commit": commit, "members": mapping},
    )
    byte_counts = sorted(
        len(render_body(normalized[row.transcript_id]).encode()) for row in assigned
    )
    family_sizes = Counter(row.family_id for row in assigned)
    counts = {
        "source_directories": len(directories),
        "source_categories": len(
            {directory.relative_to(attacks).parts[0] for directory in directories}
        ),
        "attack_transcript_files": len(list(attacks.rglob("transcript.jsonl"))),
        "benign_transcript_files": len(list(attacks.rglob("benign.jsonl"))),
        "all_jsonl_files": len(list(attacks.rglob("*.jsonl"))),
        "excluded_multisession_directories": reason_counts["multisession_or_related"],
        "excluded_multisession_cases": len(
            {member["stage_stem"] for i, member in enumerate(members) if find(i) in multi_families}
        ),
        "excluded_unpaired_directories": reason_counts["missing_paired_transcript"],
        "excluded_malformed_directories": reason_counts["invalid_official_format"],
        "eligible_pairs": len(assigned) // 2,
        "eligible_transcripts": len(assigned),
        "eligible_families": len(family_sizes),
        "related_multiple_pair_families": sum(size > 2 for size in family_sizes.values()),
        "development_pairs": sum(row.split == "development" for row in assigned) // 2,
        "test_pairs": sum(row.split == "test" for row in assigned) // 2,
        "requested_development_pairs": development_pairs,
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "data_origin": "sleight_bench",
        "dataset_commit": commit,
        "source_repository": DATASET_REPOSITORY,
        "normalized_content_digest": hashlib.sha256(
            json.dumps(input_hashes, sort_keys=True).encode()
        ).hexdigest(),
        "source_license": "MIT",
        "split_seed": seed,
        "counts": counts,
        "exclusions": exclusions,
        "family_rule": (
            "Transitive stage stems, exact normalized tasks/scenarios, sidecar references"
        ),
        "model_input_view": "Visible text and tool interactions; thinking and canaries excluded",
        "initial_task_rule": "First real user text, preserved literally outside body",
        "body_size": {
            "unit": "utf8_bytes",
            "min": min(byte_counts),
            "median": byte_counts[len(byte_counts) // 2],
            "max": max(byte_counts),
        },
        "model_token_counts": None,
        "context_window_validation": "Pending explicit model selection and provider token counting",
        "performance_scores_generated": False,
    }
    public_dir.mkdir(parents=True, exist_ok=True)
    (public_dir / "inventory.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (public_dir / "split.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["transcript_id", "scenario_id", "family_id", "split"]
        )
        writer.writeheader()
        for label in assigned:
            writer.writerow(label.model_dump(exclude={"label"}))
    _write_private_json(
        private_dir / "manifest.json",
        {
            "canaries": canary_values,
            **manifest,
            "upstream": str(upstream),
            "labels_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
            "normalized_sha256": normalized_hashes,
        },
    )
    return manifest


def load_canaries(private_dir: Path) -> list[str]:
    path = Path(private_dir) / "canaries.json"
    if not path.exists():
        raise DatasetError("Private canaries missing; acquire and inventory the real data first")
    values = json.loads(path.read_text()).get("canaries")
    if not isinstance(values, list) or not values or not all(isinstance(x, str) for x in values):
        raise DatasetError("Invalid private canary envelope")
    return values


def load_dataset(
    private_dir: Path,
    split: str | None = None,
) -> tuple[list[TranscriptInput], list[EvaluationLabel]]:
    """Load private envelopes after split and content integrity checks, with no fallback."""
    private_dir = Path(private_dir)
    manifest_path = private_dir / "manifest.json"
    labels_path = private_dir / "labels.jsonl"
    if not manifest_path.exists() or not labels_path.exists():
        raise DatasetError("Real dataset missing; run acquisition and inventory explicitly")
    manifest = json.loads(manifest_path.read_text())
    if hashlib.sha256(labels_path.read_bytes()).hexdigest() != manifest["labels_sha256"]:
        raise DatasetError("Evaluator labels changed after inventory")
    labels = [
        EvaluationLabel.model_validate(row)
        for row in _read_jsonl(labels_path)
        if "canaries" not in row
    ]
    validate_splits(labels)
    if split is not None and split not in ("development", "test"):
        raise DatasetError("Choose development or test split")
    selected = [row for row in labels if split is None or row.split == split]
    inputs = []
    expected_canaries = load_canaries(private_dir)
    for label in selected:
        path = private_dir / "normalized" / (label.transcript_id + ".json")
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["normalized_sha256"].get(
            label.transcript_id
        ):
            raise DatasetError("Normalized transcript changed after inventory")
        envelope = json.loads(path.read_text())
        if envelope.get("canaries") != expected_canaries:
            raise DatasetError("Private normalized artifact lost its source canaries")
        transcript = TranscriptInput.model_validate(envelope["input"])
        if (
            transcript.transcript_id != label.transcript_id
            or transcript.data_origin != "sleight_bench"
        ):
            raise DatasetError("Transcript identity or origin mismatch")
        inputs.append(transcript)
    return inputs, selected


def acquire_dataset(private_dir: Path) -> dict[str, Any]:
    """Clone the reviewed pin and run only the documented local decryptor.

    This is an explicit acquisition action, never run during import, tests, or model
    inference. Git and the official decryptor are the only child processes; no
    transcript or evaluator command is executed. Install the optional ``data`` extra.
    """
    import re
    import subprocess
    import sys

    private_dir = _private_dir(private_dir)
    upstream = private_dir / "upstream"
    if not upstream.exists():
        subprocess.run(
            ["git", "clone", "--no-checkout", DATASET_REPOSITORY, str(upstream)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(upstream), "checkout", "--detach", DATASET_COMMIT],
            check=True,
            capture_output=True,
            text=True,
        )
    commit = _verified_source_commit(upstream)
    trusted_script = subprocess.check_output(
        ["git", "-C", str(upstream), "show", f"{commit}:decrypt.py"]
    )
    if (upstream / "decrypt.py").read_bytes() != trusted_script:
        raise DatasetError("Official decryptor differs from the reviewed pinned source")
    readme = (upstream / "README.md").read_text()
    match = re.search(r"Decryption key: `([^`]+)`", readme)
    if match is None:
        raise DatasetError("Official README has no documented decryption key")
    log_path = private_dir / "decryption.log"
    with log_path.open("w") as handle:
        result = subprocess.run(
            [sys.executable, str(upstream / "decrypt.py"), "--key", match.group(1)],
            stdout=handle,
            stderr=handle,
            check=False,
        )
    log_path.chmod(0o600)
    if result.returncode:
        raise DatasetError(
            "Official decryption failed; inspect the private log and data dependency"
        )
    return {"dataset_commit": commit, "decryption_completed": True}
