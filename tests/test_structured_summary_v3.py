"""Synthetic checks for citation-constrained drafts and deterministic ID indexing."""

import copy
import json
from pathlib import Path

import pytest

from context_audit.representations import STRUCTURED_FIELDS, validate_summary
from context_audit.runtime_models import AuditConfig, QwenConfig
from context_audit.storage import PrivateStore, digest

CLAIM_FIELDS = (
    "environment_and_state",
    "actions_and_observed_results",
    "important_identifiers",
    "contradictions_or_missing_information",
)


def draft():
    return {
        "environment_and_state": [{"text": "Toy room inspected.", "evidence_event_ids": ["E0002"]}],
        "actions_and_observed_results": [{
            "text": "Blue blocks counted.", "evidence_event_ids": ["E0003", "E0002"],
        }],
        "important_identifiers": [{"text": "/toy/blocks.csv", "evidence_event_ids": ["E0002"]}],
        "contradictions_or_missing_information": [{
            "text": "Inventory completeness unknown.", "evidence_event_ids": ["E0003"],
        }],
    }


def test_schema_is_fresh_and_requires_each_item_to_select_visible_evidence():
    from context_audit.structured_summary import structured_summary_schema

    schema = structured_summary_schema({"E0003", "E0002"})
    assert schema["type"] == "object" and schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"]) == set(CLAIM_FIELDS)
    for field in CLAIM_FIELDS:
        field_schema = schema["properties"][field]
        assert field_schema["type"] == "array"
        item = field_schema["items"]
        assert item["type"] == "object" and item["additionalProperties"] is False
        assert set(item["required"]) == {"text", "evidence_event_ids"}
        assert item["properties"]["text"] == {"type": "string"}
        ids = item["properties"]["evidence_event_ids"]
        assert ids["type"] == "array" and ids["minItems"] == 1
        assert ids["items"] == {"type": "string", "enum": ["E0002", "E0003"]}
    schema["required"].clear()
    ids["items"]["enum"].clear()
    fresh = structured_summary_schema({"E0002", "E0003"})
    assert fresh["required"]
    assert fresh["properties"]["contradictions_or_missing_information"]["items"][
        "properties"
    ]["evidence_event_ids"]["items"]["enum"] == ["E0002", "E0003"]


def test_assembly_preserves_every_claim_and_derives_only_the_sorted_deduplicated_index():
    from context_audit.structured_summary import assemble_structured_summary

    fields = draft()
    fields["important_identifiers"].append({
        "text": "Unicode identifier café.csv\n with exact spacing. ",
        "evidence_event_ids": ["E0003", "E0003"],
    })
    original = copy.deepcopy(fields)
    raw = json.dumps(fields, ensure_ascii=True, indent=4)
    assembled = assemble_structured_summary(raw, {"E0001", "E0002", "E0003"})
    decoded = json.loads(assembled)
    assert set(decoded) == STRUCTURED_FIELDS
    assert fields == original
    for field in CLAIM_FIELDS:
        assert len(decoded[field]) == len(fields[field])
        for raw_claim, assembled_claim in zip(fields[field], decoded[field], strict=True):
            assert assembled_claim.startswith(raw_claim["text"] + " ")
    assert decoded["important_identifiers"][-1] == (
        "Unicode identifier café.csv\n with exact spacing.  [E0003]"
    )
    assert decoded["actions_and_observed_results"] == ["Blue blocks counted. [E0002] [E0003]"]
    assert decoded["source_event_ids"] == ["E0002", "E0003"]
    assert raw == json.dumps(original, ensure_ascii=True, indent=4)
    assert validate_summary(assembled, "structured_summary", 1024, len, {"E0002", "E0003"})


@pytest.mark.parametrize("field", CLAIM_FIELDS)
def test_assembly_never_invents_citations_for_uncited_claims(field):
    from context_audit.structured_summary import assemble_structured_summary

    fields = draft()
    fields[field] = [{"text": "This independent claim has no citation.", "evidence_event_ids": []}]
    raw = json.dumps(fields)
    with pytest.raises(ValueError, match="citation"):
        assemble_structured_summary(raw, {"E0002", "E0003"})
    assert json.loads(raw) == fields


@pytest.mark.parametrize("claim", [
    {"text": "Unknown observation.", "evidence_event_ids": ["E9001"]},
    {"text": "Other E9001.", "evidence_event_ids": ["E0002"]},
])
def test_assembly_rejects_unknown_event_ids(claim):
    from context_audit.structured_summary import assemble_structured_summary

    fields = draft()
    fields["important_identifiers"] = [claim]
    with pytest.raises(ValueError, match="unknown"):
        assemble_structured_summary(json.dumps(fields), {"E0002", "E0003"})


def test_assembly_rejects_text_references_not_selected_for_that_same_item():
    from context_audit.structured_summary import assemble_structured_summary

    fields = draft()
    fields["important_identifiers"] = [{
        "text": "This mentions visible E0003.", "evidence_event_ids": ["E0002"],
    }]
    with pytest.raises(ValueError, match="not selected for this item"):
        assemble_structured_summary(json.dumps(fields), {"E0002", "E0003"})


def test_assembly_rejects_an_uncited_item_even_when_adjacent_items_select_evidence():
    from context_audit.structured_summary import assemble_structured_summary

    fields = draft()
    fields["important_identifiers"].append({"text": "Missing its own evidence selection."})
    with pytest.raises(ValueError, match="claim fields do not match schema"):
        assemble_structured_summary(json.dumps(fields), {"E0002", "E0003"})


@pytest.mark.parametrize("empty", ["", " \n\t"])
def test_assembly_rejects_empty_claim_text_even_with_selected_evidence(empty):
    from context_audit.structured_summary import assemble_structured_summary

    fields = draft()
    fields["important_identifiers"][0]["text"] = empty
    with pytest.raises(ValueError, match="nonempty text"):
        assemble_structured_summary(json.dumps(fields), {"E0002", "E0003"})


def test_assembly_rejects_duplicate_keys_instead_of_silently_losing_claims():
    from context_audit.structured_summary import assemble_structured_summary

    raw = json.dumps(draft())[:-1] + ', "environment_and_state": []}'
    with pytest.raises(ValueError, match="Duplicate"):
        assemble_structured_summary(raw, {"E0002", "E0003"})


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_assembly_rejects_non_json_numeric_constants(constant):
    from context_audit.structured_summary import assemble_structured_summary

    fields = draft()
    fields["important_identifiers"] = ["CONSTANT_PLACEHOLDER"]
    raw = json.dumps(fields).replace('"CONSTANT_PLACEHOLDER"', constant)
    with pytest.raises(ValueError, match="strict JSON"):
        assemble_structured_summary(raw, {"E0002", "E0003"})


@pytest.mark.parametrize("mutation", ["extra_index", "missing_field", "not_array", "not_string"])
def test_assembly_rejects_wrong_draft_shape(mutation):
    from context_audit.structured_summary import assemble_structured_summary

    fields = draft()
    if mutation == "extra_index":
        fields["source_event_ids"] = ["E0002", "E0003"]
    elif mutation == "missing_field":
        fields.pop("important_identifiers")
    elif mutation == "not_array":
        fields["important_identifiers"] = "[E0002]"
    else:
        fields["important_identifiers"] = [{"claim": "[E0002]"}]
    with pytest.raises(ValueError):
        assemble_structured_summary(json.dumps(fields), {"E0002", "E0003"})


class SyntheticProvider:
    counter_method = "synthetic_character_count_not_model_tokens"
    canaries = ["Independent v3 synthetic private canary."]

    def __init__(self, store, outputs):
        self.store = store
        self.outputs = iter(outputs)
        self.requests = []

    def count_text(self, model, text):
        return len(text)

    def generate(self, **request):
        self.requests.append(request)
        key = f"independent-call-{len(self.requests)}"
        value = dict(status="ok", text=next(self.outputs), cost_usd=0.01)
        self.store.put("calls", key, {**value, "canaries": self.canaries})
        return key, value


def synthetic_transcript():
    from context_audit.dataset import parse_transcript

    transcript = parse_transcript(
        Path("tests/fixtures/visible_events.jsonl"), "t_" + "c" * 24,
        data_origin="synthetic_fixture",
    )
    transcript.events[-1].text = "The independent toy room contains blue blocks. " * 200
    return transcript


def config(mode="schema_citations_v1"):
    return AuditConfig(
        provider="qwen_local", qwen=QwenConfig(),
        monitor_model="Qwen/Qwen3.8-27B", summarizer_model="Qwen/Qwen3.8-27B",
        structured_summary_mode=mode,
    )


def run_summary(tmp_path, outputs, *, condition="structured_summary", mode="schema_citations_v1"):
    from context_audit.runner import make_representation

    store = PrivateStore(tmp_path)
    provider = SyntheticProvider(store, outputs)
    cfg = config(mode)
    representation, calls = make_representation(
        synthetic_transcript(), condition, cfg, provider, store, 0, 50000
    )
    return representation, calls, provider, store, cfg


def test_runner_sends_draft_schema_and_preserves_raw_and_assembled_private_artifacts(tmp_path):
    from context_audit.runner import prompts
    from context_audit.structured_summary import structured_summary_schema

    raw = json.dumps(draft(), indent=2)
    representation, calls, provider, store, cfg = run_summary(tmp_path, [raw])
    assert representation.status == "ok" and len(calls) == 1
    visible = {event.event_id for event in synthetic_transcript().events}
    assert provider.requests[0]["structured_schema"] == structured_summary_schema(visible)
    assert provider.requests[0]["max_tokens"] == 1600
    selected = Path("prompts/summary_structured_schema.txt").read_text()
    assert prompts(cfg)["summary_structured"] == selected
    assert selected in provider.requests[0]["system"]
    assert store.get("calls", "independent-call-1")["text"] == raw == calls[0]["text"]
    assembled = json.loads(representation.text)
    assert assembled["important_identifiers"] == ["/toy/blocks.csv [E0002]"]
    assert assembled["actions_and_observed_results"] == ["Blue blocks counted. [E0002] [E0003]"]
    assert assembled["source_event_ids"] == ["E0002", "E0003"]
    diagnostics = [json.loads(path.read_text()) for path in (
        store.root / "summary_validation"
    ).glob("*.json")]
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic["assembly_mode"] == "schema_citations_v1"
    assert diagnostic["raw_measured_tokens"] == len(raw)
    assert diagnostic["raw_text_hash"] == digest(raw)
    assert diagnostic["assembled_tokens"] == len(representation.text)
    assert diagnostic["assembled_text_hash"] == representation.content_hash
    saved = store.get("summary_assemblies", diagnostic["assembly_key"])
    assert saved["text"] == representation.text
    assert saved["raw_call_key"] == "independent-call-1"
    assert saved["canaries"] == diagnostic["canaries"] == provider.canaries
    assert provider.canaries[0] not in json.dumps(provider.requests)


def test_runner_checks_final_cap_after_adding_the_index_without_dropping_claims(tmp_path):
    from context_audit.runner import monitor_representation

    fields = draft()
    raw = json.dumps(fields, separators=(",", ":"))
    fields["actions_and_observed_results"][0]["text"] += "x" * (1300 - len(raw))
    raw = json.dumps(fields, separators=(",", ":"))
    assert len(raw) == 1300
    representation, calls, provider, store, cfg = run_summary(tmp_path, [raw, raw])
    assert representation.status == "budget_violation"
    assert representation.budget == 1024 and representation.text == ""
    assert len(calls) == len(provider.requests) == 2
    for path in (store.root / "summary_validation").glob("*.json"):
        diagnostic = json.loads(path.read_text())
        assert diagnostic["raw_measured_tokens"] == len(raw)
        assert diagnostic["assembled_tokens"] > 1024
        assert diagnostic["reason_code"] == "token_ceiling_exceeded"
        saved = store.get("summary_assemblies", diagnostic["assembly_key"])
        saved_fields = json.loads(saved["text"])
        for field in CLAIM_FIELDS:
            assert len(saved_fields[field]) == len(fields[field])
            assert saved_fields[field][0].startswith(fields[field][0]["text"] + " ")
    result, monitor_calls = monitor_representation(
        synthetic_transcript(), representation, cfg, provider, 0, 50000
    )
    assert monitor_calls == [] and result.suspicion_score is None and result.escalate


def test_cap_applies_to_the_final_representation_rather_than_verbose_draft_objects(tmp_path):
    fields = draft()
    raw = json.dumps(fields, separators=(",", ":"))
    fields["actions_and_observed_results"][0]["text"] += "x" * (1100 - len(raw))
    raw = json.dumps(fields, separators=(",", ":"))
    representation, calls, _, store, _ = run_summary(tmp_path, [raw])
    assert representation.status == "ok" and len(calls) == 1
    assert len(raw) > representation.budget >= representation.measured_tokens
    diagnostic = next((store.root / "summary_validation").glob("*.json"))
    measured = json.loads(diagnostic.read_text())
    assert measured["raw_measured_tokens"] == len(raw)
    assert measured["measured_tokens"] == measured["assembled_tokens"] == len(representation.text)


def test_runner_rejects_uncited_drafts_twice_without_inventing_evidence(tmp_path):
    fields = draft()
    fields["important_identifiers"] = [{
        "text": "SYNTHETIC_UNCITED_CANDIDATE", "evidence_event_ids": [],
    }]
    raw = json.dumps(fields)
    representation, calls, provider, store, _ = run_summary(tmp_path, [raw, raw])
    assert representation.status == "invalid_output" and representation.text == ""
    assert len(calls) == len(provider.requests) == 2
    assert "SYNTHETIC_UNCITED_CANDIDATE" not in provider.requests[1]["text"]
    assert not (store.root / "summary_assemblies").exists()
    for path in (store.root / "summary_validation").glob("*.json"):
        diagnostic = json.loads(path.read_text())
        assert diagnostic["reason_code"] == "uncited_structured_claim"
        assert diagnostic["assembled_tokens"] is None


@pytest.mark.parametrize("condition, mode", [
    ("free_summary", "schema_citations_v1"), ("structured_summary", "prompt"),
])
def test_legacy_and_free_summary_requests_do_not_receive_structured_constraints(
    tmp_path, condition, mode
):
    fields = {field: ["Observed toy data [E0002]."] for field in CLAIM_FIELDS}
    fields["source_event_ids"] = ["E0002"]
    output = "Toy blocks counted [E0003]." if condition == "free_summary" else json.dumps(fields)
    representation, _, provider, _, _ = run_summary(
        tmp_path, [output], condition=condition, mode=mode
    )
    assert representation.status == "ok" and representation.text == output
    assert "structured_schema" not in provider.requests[0]


def test_monitor_does_not_receive_draft_schema_or_the_unassembled_draft(tmp_path):
    from context_audit.runner import monitor_representation

    raw = json.dumps(draft())
    representation, _, provider, _, cfg = run_summary(tmp_path, [raw])
    provider.outputs = iter([json.dumps(dict(
        suspicion_score=25, brief_justification="Observed toy data.",
        evidence_event_ids=["E0003"], uncertainties=[],
    ))])
    result, calls = monitor_representation(
        synthetic_transcript(), representation, cfg, provider, 0, 50000
    )
    assert result.status == "ok" and len(calls) == 1
    assert "structured_schema" not in provider.requests[-1]
    assert representation.text in provider.requests[-1]["text"]
    assert raw not in provider.requests[-1]["text"]


def test_identity_shortcut_never_generates_or_assembles_a_draft(tmp_path):
    from context_audit.runner import make_representation
    from context_audit.schemas import Event, TranscriptInput

    source = TranscriptInput(
        transcript_id="t_" + "d" * 24, initial_task="Count toy blocks.",
        events=[Event(event_id="E0001", kind="text", role="assistant", cwd="/toy", text="OK.")],
        data_origin="synthetic_fixture",
    )
    store = PrivateStore(tmp_path)
    provider = SyntheticProvider(store, [])
    representation, calls = make_representation(
        source, "structured_summary", config(), provider, store, 0, 50000
    )
    assert representation.status == "ok" and not representation.compression_applied
    assert calls == provider.requests == []
    assert not (store.root / "summary_assemblies").exists()


def test_preflight_counts_the_selected_draft_prompt_before_any_generation(tmp_path, monkeypatch):
    import context_audit.dataset as dataset
    import context_audit.qwen_provider as qwen_provider
    import context_audit.runner as runner

    selected = Path("prompts/summary_structured_schema.txt").read_text()

    class InventoryBoundary(SyntheticProvider):
        counted_prompts = []

        def model_info(self, model):
            return {"id": model, "max_input_tokens": 2400}

        def count_request(self, model, system, text):
            self.counted_prompts.append(system)
            return 900 if selected in system else 100

        def generate(self, **request):
            pytest.fail("Selected draft prompt exceeded preflight; must not generate")

    source = synthetic_transcript()
    provider = InventoryBoundary(PrivateStore(tmp_path), [])
    cfg = config().model_copy(update={
        "run_dir": str(tmp_path), "qwen": QwenConfig(
            model_revision="1" * 40, gpu_budget_hours=12,
        ),
    })
    monkeypatch.setattr(qwen_provider, "QwenProvider", lambda *args, **kwargs: provider)
    monkeypatch.setattr(dataset, "load_dataset", lambda *args, **kwargs: ([source], []))
    monkeypatch.setattr(runner, "code_state", lambda: {"commit": "a" * 40, "hash": "synthetic"})
    with pytest.raises(ValueError, match="transcripts exceed context"):
        runner._run_locked(
            cfg, [source], {}, {"dataset_commit": "b" * 40},
            {"code_hash": "synthetic", "prompts": {}}, None, [], {}, {},
        )
    assert sum(selected in system for system in provider.counted_prompts) == 2
    inventory = json.loads((tmp_path / "manifests/preflight.json").read_text())
    assert inventory["items"][0]["summary_input_tokens"] == 900
