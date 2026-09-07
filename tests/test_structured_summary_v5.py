"""Independent fixtures for bounded drafts; no benchmark content or live generation."""

import copy
import json

import pytest
from test_structured_summary_v3 import SyntheticProvider, config, draft, synthetic_transcript

from context_audit.render import render_body
from context_audit.runner import make_representation, monitor_representation
from context_audit.storage import PrivateStore
from context_audit.structured_summary import (
    CLAIM_FIELDS,
    assemble_structured_summary,
    structured_summary_schema,
)


@pytest.mark.parametrize("budget", [1024, 1536, 2048])
def test_bounded_schema_constrains_each_item_without_changing_the_legacy_schema(budget):
    schema = structured_summary_schema({"E0002", "E0003"}, token_budget=budget)
    for field in CLAIM_FIELDS:
        array = schema["properties"][field]
        assert array["maxItems"] == 2
        properties = array["items"]["properties"]
        assert properties["evidence_event_ids"]["maxItems"] == 2
        assert f"{{0,{budget // 4}}}" in properties["text"]["pattern"]
        assert "maxLength" not in properties["text"]
    legacy = structured_summary_schema({"E0002", "E0003"})
    assert "maxItems" not in legacy["properties"][CLAIM_FIELDS[0]]
    assert legacy["properties"][CLAIM_FIELDS[0]]["items"]["properties"]["text"] == {
        "type": "string"
    }


@pytest.mark.parametrize("mutation", ["claims", "references", "characters"])
def test_local_guard_rejects_excess_without_removing_claims_or_rewriting_text(mutation):
    fields = draft()
    claim = fields[CLAIM_FIELDS[0]][0]
    if mutation == "claims":
        fields[CLAIM_FIELDS[0]] = [copy.deepcopy(claim) for _ in range(3)]
    elif mutation == "references":
        claim["evidence_event_ids"] = ["E0002"] * 3
    else:
        claim["text"] = "x" * 257
    raw = json.dumps(fields)
    with pytest.raises(ValueError, match="bounded draft"):
        assemble_structured_summary(raw, {"E0002", "E0003"}, token_budget=1024)
    assert json.dumps(fields) == raw
    # This amendment must not silently change validation of earlier versions.
    assert assemble_structured_summary(raw, {"E0002", "E0003"})


@pytest.mark.parametrize("ensure_ascii", [False, True])
def test_bounded_assembly_preserves_unicode_escapes_spacing_and_selected_citations(ensure_ascii):
    fields = draft()
    original = 'Exact "quote", \\ path, café\n雪 🧪. '
    fields[CLAIM_FIELDS[0]][0]["text"] = original
    raw = json.dumps(fields, ensure_ascii=ensure_ascii)
    final = json.loads(assemble_structured_summary(
        raw, {"E0002", "E0003"}, token_budget=1024,
    ))
    assert final[CLAIM_FIELDS[0]] == [original + " [E0002]"]


def test_local_limit_counts_original_json_escapes_like_the_pinned_decoder():
    fields = draft()
    fields[CLAIM_FIELDS[0]][0]["text"] = "🧪" * 256
    # Literal supplementary characters take one unit; escaped surrogate pairs two.
    assert assemble_structured_summary(
        json.dumps(fields, ensure_ascii=False), {"E0002", "E0003"}, token_budget=1024,
    )
    with pytest.raises(ValueError, match="bounded draft"):
        assemble_structured_summary(
            json.dumps(fields, ensure_ascii=True), {"E0002", "E0003"}, token_budget=1024,
        )
    fields[CLAIM_FIELDS[0]][0]["text"] = "🧪" * 128
    assert assemble_structured_summary(
        json.dumps(fields, ensure_ascii=True), {"E0002", "E0003"}, token_budget=1024,
    )


def test_lexical_guard_distinguishes_escaped_text_inside_claim_from_json_keys():
    fields = draft()
    original = 'Literal fragment: "text": "keep it"; \\ and 🧪.'
    fields[CLAIM_FIELDS[0]][0]["text"] = original
    raw = json.dumps(fields).replace('"text":', '"\\u0074ext":')
    final = json.loads(assemble_structured_summary(
        raw, {"E0002", "E0003"}, token_budget=1024,
    ))
    assert final[CLAIM_FIELDS[0]][0] == original + " [E0002]"


def test_bounded_assembly_reports_unpaired_surrogate_as_validation_failure():
    fields = draft()
    fields[CLAIM_FIELDS[0]][0]["text"] = "Invalid \ud800 character"
    with pytest.raises(ValueError, match="invalid Unicode"):
        assemble_structured_summary(
            json.dumps(fields), {"E0002", "E0003"}, token_budget=1024,
        )


def bounded_config():
    return config("schema_citations_bounded_v1").model_copy(update={
        "token_minimum": 1024, "token_maximum": 2048, "summary_max_tokens": 3200,
    })


def test_runner_supplies_bounds_on_both_attempts_and_never_sends_rejected_draft(tmp_path):
    fields = draft()
    fields[CLAIM_FIELDS[0]][0]["text"] = "SYNTHETIC_REJECTED_" * 40
    rejected = json.dumps(fields)
    accepted = json.dumps(draft())
    store = PrivateStore(tmp_path)
    provider = SyntheticProvider(store, [rejected, accepted])
    cfg = bounded_config()
    rep, calls = make_representation(
        synthetic_transcript(), "structured_summary", cfg, provider, store, 0, 196608,
    )
    assert rep.status == "ok" and len(calls) == 2
    assert "SYNTHETIC_REJECTED_" not in json.dumps(provider.requests)
    for request in provider.requests:
        assert request["max_tokens"] == 3200
        assert "512" in request["text"] and "at most 2" in request["text"]
        assert request["structured_schema"]["properties"][CLAIM_FIELDS[0]]["maxItems"] == 2
    diagnostics = [
        json.loads(p.read_text()) for p in (tmp_path / "summary_validation").glob("*.json")
    ]
    assert {d["reason_code"] for d in diagnostics} == {"draft_length_bounds_exceeded", "valid"}
    assert all(d["assembly_mode"] == "schema_citations_bounded_v1" for d in diagnostics)


def test_structural_bounds_do_not_replace_final_exact_token_validation(tmp_path):
    # Every item meets its character/reference limit, yet the final JSON exceeds
    # this deliberately pessimistic synthetic counter. It must fail, not be cut.
    fields = {field: [{"text": "x" * 500, "evidence_event_ids": ["E0002"]}]
              for field in CLAIM_FIELDS}
    store = PrivateStore(tmp_path)
    provider = SyntheticProvider(store, [json.dumps(fields)] * 2)
    cfg, transcript = bounded_config(), synthetic_transcript()
    rep, calls = make_representation(
        transcript, "structured_summary", cfg, provider, store, 0, 196608,
    )
    assert rep.status == "budget_violation" and rep.text == "" and len(calls) == 2
    monitor, monitor_calls = monitor_representation(transcript, rep, cfg, provider, 0, 196608)
    assert monitor.suspicion_score is None and monitor.escalate and not monitor_calls


@pytest.mark.parametrize("condition", ["head_tail", "free_summary", "structured_summary"])
def test_short_body_uses_common_floor_and_identity_below_floor(tmp_path, condition):
    store = PrivateStore(tmp_path)
    provider = SyntheticProvider(store, [json.dumps(draft()) if condition == "structured_summary"
                                         else "Toy observation [E0002]"])
    cfg, transcript = bounded_config(), synthetic_transcript()
    transcript.events[-1].text = "Toy " * 400
    rep, _ = make_representation(transcript, condition, cfg, provider, store, 0, 196608)
    assert 1024 < len(render_body(transcript)) < 4096
    assert rep.budget == 1024 and rep.status == "ok"
    transcript.events[-1].text = "Toy"
    short, calls = make_representation(transcript, condition, cfg, provider, store, 1, 196608)
    assert short.text == render_body(transcript) and not calls
    assert not short.compression_applied


def test_full_history_stays_integral_in_bounded_mode(tmp_path):
    store = PrivateStore(tmp_path)
    provider = SyntheticProvider(store, [])
    transcript = synthetic_transcript()
    rep, calls = make_representation(
        transcript, "full", bounded_config(), provider, store, 0, 196608,
    )
    assert rep.text == render_body(transcript) and not calls


def test_preflight_counts_bounded_initial_and_retry_requests_before_generation(
    tmp_path, monkeypatch,
):
    import context_audit.dataset as dataset
    import context_audit.qwen_provider as qwen_provider
    import context_audit.runner as runner
    from context_audit.runtime_models import QwenConfig

    class InventoryBoundary(SyntheticProvider):
        requests_counted = []

        def model_info(self, model):
            return {"id": model, "max_input_tokens": 4000}

        def count_request(self, model, system, text):
            self.requests_counted.append(text)
            return 900 if "Decoder limits:" in text else 100

        def generate(self, **request):
            pytest.fail("Bounded requests exceed context and must not generate")

    source = synthetic_transcript()
    provider = InventoryBoundary(PrivateStore(tmp_path), [])
    cfg = bounded_config().model_copy(update={
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
    bounded_requests = [r for r in provider.requests_counted if "Decoder limits:" in r]
    assert len(bounded_requests) == 2
    assert all("512" in r for r in bounded_requests)
    assert sum("Regenerate once" in r for r in bounded_requests) == 1
    inventory = json.loads((tmp_path / "manifests/preflight.json").read_text())
    assert inventory["items"][0]["summary_input_tokens"] == 900
