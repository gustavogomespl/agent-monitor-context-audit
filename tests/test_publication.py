import json

from context_audit.sanitation import scan_public_files


def test_scanner_detects_private_text_embedded_in_notebook_outputs(tmp_path):
    text = "Example independent benchmark evidence " + "identifiers and observed changes " * 8
    source = tmp_path / "private" / "fixture.jsonl"
    source.parent.mkdir()
    source.write_text(json.dumps({"content": {"type": "text", "text": text}}))
    notebook = tmp_path / "analysis.ipynb"
    notebook.write_text(json.dumps({"cells": [dict(cell_type="code", outputs=[dict(text=text)])]}))
    findings = scan_public_files([notebook], [source], root=tmp_path)
    assert any(f["reason"] == "protected_content_match" for f in findings)
    assert any(f["reason"] == "notebook_outputs_present" for f in findings)
    assert text not in json.dumps(findings)


def test_scanner_rejects_tracked_private_paths_even_if_content_unrecognized(tmp_path):
    path = tmp_path / "data/private/unknown.txt"
    path.parent.mkdir(parents=True)
    path.write_text("short")
    assert scan_public_files([path], [], root=tmp_path)[0]["reason"] == "private_path"


def test_scanner_accepts_clean_numeric_artifacts(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text("transcript_id,label,suspicion_score\nt_opaque,1,72\n")
    assert scan_public_files([path], [], root=tmp_path) == []


def test_scanner_detects_canary_identifier_shorter_than_normal_window(tmp_path):
    identifier = "01234567-89ab-cdef-0123-456789abcdef"
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"canary": "canary GUID " + identifier}))
    leaked = tmp_path / "public.md"
    leaked.write_text("Copied identifier: " + identifier)
    findings = scan_public_files([leaked], [source], root=tmp_path)
    assert any(item["reason"] == "source_canary_match" for item in findings)
    assert identifier not in json.dumps(findings)
