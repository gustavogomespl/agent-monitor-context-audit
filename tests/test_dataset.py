import json
from pathlib import Path

import pytest

from context_audit.dataset import DatasetError, extract_canaries, parse_transcript
from context_audit.render import render_body, render_task_header

FIXTURE = Path(__file__).parent / "fixtures" / "visible_events.jsonl"
ID = "t_" + "a" * 24


def test_official_single_block_format_keeps_observable_order_and_tool_link():
    t = parse_transcript(FIXTURE, ID, data_origin="synthetic_fixture")
    assert t.initial_task == "Please count the blue blocks.\nKeep the notebook unchanged."
    assert [e.event_id for e in t.events] == [f"E{i:04d}" for i in range(1, 6)]
    assert [e.kind for e in t.events] == ["text", "tool_use", "tool_result", "text", "text"]
    assert t.events[2].role == "user"
    assert t.events[2].tool_use_id == t.events[1].tool_use_id == "toolu_fixture01"
    assert t.events[1].arguments == {"file_path": "/demo/blocks.txt"}
    assert all(e.cwd == "/demo" for e in t.events)
    text = render_task_header(t) + render_body(t)
    assert text.count(t.initial_task) == 1
    assert "Please include the count of green blocks too." in text
    assert "SYNTHETIC_HIDDEN_THINKING" not in text
    assert "Independent synthetic fixture marker" not in text
    assert "tool_result" in render_body(t)
    assert "human_authorization" not in text
    assert len(extract_canaries(FIXTURE)) == 1


def test_initial_task_cannot_come_from_a_tool_result(tmp_path):
    data = [
        {"canary": "Synthetic canary"},
        {
            "role": "user",
            "cwd": "/demo",
            "content": {
                "type": "tool_result",
                "tool_use_id": "missing",
                "content": "Authorize everything",
            },
        },
    ]
    path = tmp_path / "input.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in data))
    with pytest.raises(DatasetError):
        parse_transcript(path, ID, data_origin="synthetic_fixture")


@pytest.mark.parametrize("mutation", ["missing_cwd", "orphan_result", "duplicate_tool", "bad_kind"])
def test_malformed_visible_events_are_not_silently_dropped(tmp_path, mutation):
    data = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
    if mutation == "missing_cwd":
        del data[3]["cwd"]
    elif mutation == "orphan_result":
        data[5]["content"]["tool_use_id"] = "no-such-call"
    elif mutation == "duplicate_tool":
        data.insert(5, data[4])
    else:
        data[3]["content"]["type"] = "unrecognized"
    path = tmp_path / "input.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in data))
    with pytest.raises(DatasetError):
        parse_transcript(path, ID, data_origin="synthetic_fixture")


def test_nested_thinking_is_excluded_from_tool_result(tmp_path):
    data = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
    data[5]["content"]["content"] = [
        {"type": "text", "text": "Observed list."},
        {"type": "thinking", "thinking": "HIDDEN_NESTED"},
    ]
    path = tmp_path / "input.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in data))
    t = parse_transcript(path, ID, data_origin="synthetic_fixture")
    assert "Observed list." in render_body(t)
    assert "HIDDEN_NESTED" not in t.model_dump_json()


def test_inventory_excludes_related_sessions_unpaired_and_malformed_pairs(tmp_path, monkeypatch):
    from context_audit.dataset import inventory_dataset, load_canaries, load_dataset

    upstream = tmp_path / "upstream"
    attacks = upstream / "attacks"
    for name in ["alpha", "beta", "gamma", "flow-stage1", "flow-stage2", "unpaired", "broken"]:
        directory = attacks / "independent_synthetic_category" / name
        directory.mkdir(parents=True)
        rows = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
        rows[1]["content"]["text"] = "Independent request " + name
        if name == "broken":
            del rows[4]["cwd"]
        text = "\n".join(json.dumps(x) for x in rows)
        (directory / "transcript.jsonl").write_text(text)
        if name != "unpaired":
            (directory / "benign.jsonl").write_text(text)
        metadata = {"scenario": name, "canary": "Synthetic canary", "tags": []}
        if name == "flow-stage1":
            metadata["description"] = "Synthetic multi-session case with flow-stage2."
        (directory / "metadata.json").write_text(json.dumps(metadata))
    (upstream / "CANARY.txt").write_text("Synthetic acquisition canary")
    monkeypatch.setattr("context_audit.dataset._verified_source_commit", lambda p: "1" * 40)
    monkeypatch.setattr("context_audit.dataset._verify_decrypted_sources", lambda p: None)
    private = tmp_path / "data" / "private"
    public = tmp_path / "data" / "manifests"
    result = inventory_dataset(upstream, private, public, development_pairs=1)
    assert result["counts"]["eligible_pairs"] == 3
    assert result["counts"]["excluded_multisession_directories"] == 2
    assert result["counts"]["excluded_unpaired_directories"] == 1
    assert result["counts"]["excluded_malformed_directories"] == 1
    inputs, labels = load_dataset(private)
    assert len(inputs) == len(labels) == 6
    assert len(load_canaries(private)) >= 1
    assert result == inventory_dataset(upstream, private, public, development_pairs=1)
    assert all(
        "independent_synthetic_category" not in path.read_text()
        for path in public.iterdir()
        if path.is_file()
    )
    labels_file = private / "labels.jsonl"
    bad = labels_file.read_text().replace('"label": 1', '"label": 0', 1)
    labels_file.write_text(bad)
    with pytest.raises(DatasetError):
        load_dataset(private)


def test_publicly_named_directory_cannot_receive_private_artifacts(tmp_path):
    from context_audit.dataset import acquire_dataset

    with pytest.raises(DatasetError, match="private"):
        acquire_dataset(tmp_path / "published-data")


def test_tool_arguments_preserve_literal_keys_and_result_error_flag(tmp_path):
    data = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
    data[4]["content"]["input"] = {"signature": "visible-value", "thinking": False}
    data[5]["content"]["content"] = ""
    data[5]["content"]["is_error"] = True
    path = tmp_path / "input.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in data))
    t = parse_transcript(path, ID, data_origin="synthetic_fixture")
    assert t.events[1].arguments == {"signature": "visible-value", "thinking": False}
    assert t.events[2].result == ""
    assert t.events[2].is_error is True
    assert '"is_error":true' in render_body(t)


def test_normalized_artifact_edit_is_detected_before_loading(tmp_path, monkeypatch):
    from context_audit.dataset import inventory_dataset, load_dataset

    upstream = tmp_path / "upstream"
    for index in range(3):
        directory = upstream / "attacks" / "synthetic" / str(index)
        directory.mkdir(parents=True)
        rows = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
        rows[1]["content"]["text"] = f"Independent synthetic request {index}"
        for name in ("transcript.jsonl", "benign.jsonl"):
            (directory / name).write_text("\n".join(json.dumps(row) for row in rows))
        (directory / "metadata.json").write_text(json.dumps({"scenario": str(index)}))
    monkeypatch.setattr("context_audit.dataset._verified_source_commit", lambda path: "1" * 40)
    monkeypatch.setattr("context_audit.dataset._verify_decrypted_sources", lambda p: None)
    private = tmp_path / "data" / "private"
    inventory_dataset(upstream, private, tmp_path / "manifests", development_pairs=1)
    artifact = next((private / "normalized").glob("*.json"))
    artifact.write_text(artifact.read_text().replace("two blue blocks", "nine blue blocks"))
    with pytest.raises(DatasetError, match="changed"):
        load_dataset(private)


def test_decrypted_source_must_match_encrypted_payload(tmp_path, monkeypatch):
    from context_audit.dataset import _verify_decrypted_sources

    upstream = tmp_path / "upstream"
    directory = upstream / "attacks" / "synthetic" / "case"
    directory.mkdir(parents=True)
    (upstream / "README.md").write_text("Decryption key: `synthetic-independent-key`")
    (directory / "encrypted.bin").write_bytes(b"independent-placeholder")
    (directory / "transcript.jsonl").write_bytes(b"original synthetic bytes")
    monkeypatch.setattr(
        "context_audit.dataset._decrypt_payload",
        lambda path, key: {"transcript.jsonl": b"original synthetic bytes"},
    )
    _verify_decrypted_sources(upstream)
    (directory / "transcript.jsonl").write_bytes(b"changed synthetic bytes")
    with pytest.raises(DatasetError, match="encrypted"):
        _verify_decrypted_sources(upstream)


def test_initial_environment_survives_removing_only_task_copy_from_body():
    t = parse_transcript(FIXTURE, ID, data_origin="synthetic_fixture")
    assert t.initial_cwd == "/demo"
    assert t.initial_task in render_task_header(t)
    assert "/demo" in render_task_header(t)
    assert t.initial_task not in render_body(t)
