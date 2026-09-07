"""Independent synthetic HTTP boundaries for local Qwen inference; no GPU calls."""

import copy
import importlib
import json
from types import SimpleNamespace

import httpx
import pytest

from context_audit.provider import BudgetLedger, TokenCountError
from context_audit.storage import PrivateStore

MODEL = "Qwen/Qwen3.8-27B"
HTTPX_CLIENT = httpx.Client


def configuration(**overrides):
    from context_audit.runtime_models import QwenConfig

    return QwenConfig(
        **{
            "model_revision": "a" * 40,
            "vllm_version": "0.28.0",
            "gpu_hourly_rate_usd": 3.6,
            **overrides,
        }
    )


def response_body():
    return {
        "id": "chatcmpl-independent-fixture",
        "object": "chat.completion",
        "created": 1,
        "model": MODEL,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant", "content": "Observed result [E0003].",
                "reasoning": None, "refusal": None, "tool_calls": [],
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27},
    }


def setup_provider(
    tmp_path, monkeypatch, *, body=None, handler=None, config=None,
    ledger_unit="usd", ledger_limit=1, **kwargs,
):
    assert importlib.util.find_spec("context_audit.qwen_provider") is not None, (
        "The local Qwen provider is not implemented"
    )
    module = importlib.import_module("context_audit.qwen_provider")
    config = config or configuration()
    requests = []

    def serve(request):
        payload = json.loads(request.content) if request.content else None
        requests.append((request, payload))
        if handler:
            result = handler(request, payload)
            if result is not None:
                return result
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": config.vllm_version})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{
                "id": MODEL, "object": "model", "root": MODEL,
                "max_model_len": config.max_model_len,
            }]})
        if request.url.path == "/tokenize":
            count = 20 if "messages" in payload else 4
            return httpx.Response(200, json={
                "count": count, "max_model_len": config.max_model_len,
                "tokens": list(range(count)), "token_strs": None,
            })
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json=copy.deepcopy(body or response_body()))
        pytest.fail(f"Unexpected HTTP boundary: {request.url.path}")

    def client(**settings):
        assert settings["trust_env"] is False
        assert settings["follow_redirects"] is False
        return HTTPX_CLIENT(**settings, transport=httpx.MockTransport(serve))

    monkeypatch.setattr(module.httpx, "Client", client)
    store = PrivateStore(tmp_path)
    ledger = BudgetLedger(store, ledger_limit, unit=ledger_unit)
    provider = module.QwenProvider(
        store, ledger, config, timeout=30, canaries=["independent-source-notice"], **kwargs
    )
    return provider, store, ledger, requests


def generate(provider, **overrides):
    return provider.generate(**{
        "model": MODEL,
        "system": "Independent fixed instruction.",
        "text": "Independent transcript [E0003].",
        "max_tokens": 50,
        "context_window": 65536,
        "identity": {"phase": "monitor", "repetition": 0, "attempt": 0},
        **overrides,
    })


def test_counts_and_generation_share_template_without_tools_or_notices(tmp_path, monkeypatch):
    provider, store, ledger, requests = setup_provider(tmp_path, monkeypatch)
    assert provider.model_info(MODEL)["max_input_tokens"] == 65536
    assert provider.count_text(MODEL, "Four body tokens.") == 4
    key, record = generate(provider)
    assert record["status"] == "ok" and record["text"] == "Observed result [E0003]."
    assert record["usage"]["input_tokens"] == 20
    assert record["usage"]["output_tokens"] == 7
    assert record["cost_is_upper_bound"] is False and key in ledger.settled
    body_count = next(p for r, p in requests if r.url.path == "/tokenize" and "prompt" in p)
    assert body_count["prompt"] == "Four body tokens." and not body_count["add_special_tokens"]
    chat_count = next(p for r, p in requests if r.url.path == "/tokenize" and "messages" in p)
    generation = next(p for r, p in requests if r.url.path == "/v1/chat/completions")
    assert chat_count["messages"] == generation["messages"]
    assert chat_count["add_generation_prompt"] is generation["add_generation_prompt"] is True
    assert chat_count["chat_template_kwargs"] == generation["chat_template_kwargs"] == {
        "enable_thinking": False, "preserve_thinking": False,
    }
    assert all("authorization" not in r.headers for r, _ in requests)
    assert "tools" not in generation and "tool_choice" not in generation
    assert "independent-source-notice" not in json.dumps(generation)
    assert store.get("calls", key)["canaries"] == ["independent-source-notice"]
    assert "raw_response" not in record


@pytest.mark.parametrize("endpoint,payload", [
    ("/version", {"version": "wrong-version"}),
    ("/v1/models", {"data": [{"id": "wrong-model", "max_model_len": 65536}]}),
    ("/v1/models", {"data": [{"id": MODEL, "max_model_len": 32768}]}),
])
def test_server_mismatch_stops_before_tokenization(tmp_path, monkeypatch, endpoint, payload):
    def handler(request, _):
        if request.url.path == endpoint:
            return httpx.Response(200, json=payload)

    provider, _, ledger, requests = setup_provider(tmp_path, monkeypatch, handler=handler)
    with pytest.raises(ValueError, match="server|Server|version|model"):
        provider.model_info(MODEL)
    assert ledger.committed == 0
    assert all(r.url.path not in ("/tokenize", "/v1/chat/completions") for r, _ in requests)


def test_full_context_overflow_never_generates_or_reserves(tmp_path, monkeypatch):
    provider, _, ledger, requests = setup_provider(tmp_path, monkeypatch)
    _, record = generate(provider, context_window=60)
    assert record["status"] == "context_limit" and record["text"] == ""
    assert ledger.committed == 0
    assert all(r.url.path != "/v1/chat/completions" for r, _ in requests)


@pytest.mark.parametrize("mutation,error_status,cause", [
    ({"reasoning": "PRIVATE-THINKING-MUST-NOT-PERSIST"},
     "invalid_output", "reasoning_content_present"),
    ({"reasoning_content": "PRIVATE-THINKING-MUST-NOT-PERSIST"},
     "invalid_output", "reasoning_content_present"),
    ({"content": "<think>PRIVATE-THINKING-MUST-NOT-PERSIST</think>answer"},
     "invalid_output", "thinking_tag_present"),
    ({"tool_calls": [{"function": {"name": "independent_tool"}}]},
     "invalid_output", "tool_call_present"),
    ({"refusal": "PRIVATE-THINKING-MUST-NOT-PERSIST", "content": None},
     "refusal", "refusal_present"),
    ({"content": [{"type": "text", "text": "unexpected block"}]},
     "invalid_output", "non_text_content"),
])
def test_non_text_and_thinking_responses_are_failed_and_not_retained(
    tmp_path, monkeypatch, mutation, error_status, cause
):
    body = response_body()
    body["choices"][0]["message"].update(mutation)
    provider, store, ledger, _ = setup_provider(tmp_path, monkeypatch, body=body)
    key, record = generate(provider)
    assert record["status"] == error_status and record["text"] == ""
    assert record["diagnostic_cause"] == cause
    assert record["finish_reason"] == record["stop_reason"] == "stop"
    assert key in ledger.settled
    assert "PRIVATE-THINKING-MUST-NOT-PERSIST" not in store.path("calls", key).read_text()


@pytest.mark.parametrize("reason,status", [
    ("length", "invalid_output"), ("tool_calls", "invalid_output"),
    ("content_filter", "refusal"), ("unexpected", "invalid_output"),
])
def test_nonstop_finish_is_explicit_failure(tmp_path, monkeypatch, reason, status):
    body = response_body()
    body["choices"][0]["finish_reason"] = reason
    provider, _, _, _ = setup_provider(tmp_path, monkeypatch, body=body)
    assert generate(provider)[1]["status"] == status


@pytest.mark.parametrize("usage", [
    {}, {"prompt_tokens": True, "completion_tokens": 7, "total_tokens": 8},
    {"prompt_tokens": 20, "completion_tokens": -1, "total_tokens": 19},
    {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 99},
    {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27,
     "completion_tokens_details": {"reasoning_tokens": 1}},
    {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27,
     "completion_tokens_details": {"reasoning_tokens": False}},
])
def test_malformed_or_reasoning_usage_is_not_success(tmp_path, monkeypatch, usage):
    body = response_body()
    body["usage"] = usage
    provider, _, ledger, _ = setup_provider(tmp_path, monkeypatch, body=body)
    key, record = generate(provider)
    assert record["status"] == "invalid_output" and record["text"] == ""
    assert record["cost_is_upper_bound"] is False and key in ledger.settled


def test_timeout_reservation_persists_without_replay(tmp_path, monkeypatch):
    def handler(request, _):
        if request.url.path == "/v1/chat/completions":
            raise httpx.ReadTimeout("PRIVATE-ERROR-TEXT", request=request)

    provider, store, ledger, requests = setup_provider(tmp_path, monkeypatch, handler=handler)
    key, record = generate(provider)
    assert record["status"] == "api_error" and record["cost_is_upper_bound"] is True
    assert ledger.committed == pytest.approx(0.03) and key not in ledger.settled
    assert "PRIVATE-ERROR-TEXT" not in store.path("calls", key).read_text()
    assert generate(provider) == (key, record)
    assert sum(r.url.path == "/v1/chat/completions" for r, _ in requests) == 1


def test_cache_preserves_cost_and_repeat_attempt_identity(tmp_path, monkeypatch):
    provider, _, ledger, requests = setup_provider(tmp_path, monkeypatch)
    key, record = generate(provider)
    cost = ledger.committed
    assert generate(provider) == (key, record) and ledger.committed == cost
    repeat_key, _ = generate(provider, identity={"phase": "monitor", "repetition": 1, "attempt": 0})
    retry_key, _ = generate(provider, identity={"phase": "monitor", "repetition": 0, "attempt": 1})
    assert len({key, repeat_key, retry_key}) == 3
    seeds = [p["seed"] for r, p in requests if r.url.path == "/v1/chat/completions"]
    assert len(set(seeds)) == 3


def test_count_failure_is_explicit_and_nonbillable(tmp_path, monkeypatch):
    def handler(request, _):
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": "20", "max_model_len": 65536})

    provider, _, ledger, _ = setup_provider(tmp_path, monkeypatch, handler=handler)
    with pytest.raises(TokenCountError):
        provider.count_text(MODEL, "Independent fixture")
    _, record = generate(provider)
    assert record["status"] == "api_error" and record["error"] == "token_count_failed"
    assert ledger.committed == 0


def test_maximum_calls_and_financial_cap_prevent_generation(tmp_path, monkeypatch):
    provider, _, ledger, requests = setup_provider(tmp_path, monkeypatch, max_calls=1)
    generate(provider)
    _, second = generate(provider, identity={"attempt": 1})
    assert second["status"] == "budget_violation"
    assert sum(r.url.path == "/v1/chat/completions" for r, _ in requests) == 1
    provider.max_calls = 10
    ledger.limit = ledger.committed + 0.001
    _, third = generate(provider, identity={"attempt": 2})
    assert third["status"] == "budget_violation"
    assert sum(r.url.path == "/v1/chat/completions" for r, _ in requests) == 1


def test_elapsed_gpu_cost_is_settled_and_restored_after_restart(tmp_path, monkeypatch):
    clock = SimpleNamespace(value=10.0)

    def handler(request, _):
        if request.url.path == "/v1/chat/completions":
            clock.value += 2

    provider, store, ledger, _ = setup_provider(tmp_path, monkeypatch, handler=handler)
    module = importlib.import_module("context_audit.qwen_provider")
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock.value))
    key, record = generate(provider)
    assert record["latency_seconds"] == 2
    assert record["cost_usd"] == pytest.approx(0.002)
    assert store.read_journal("budget")[0]["amount"] == pytest.approx(0.03)
    assert ledger.committed == pytest.approx(0.002)
    restarted, _, restarted_ledger, requests = setup_provider(tmp_path, monkeypatch)
    assert generate(restarted) == (key, record)
    assert restarted_ledger.committed == pytest.approx(0.002) and not requests


def test_saved_response_recovers_interrupted_settlement_without_replay(tmp_path, monkeypatch):
    provider, store, ledger, requests = setup_provider(tmp_path, monkeypatch)
    settle = ledger.settle

    def fail_settlement(*_):
        raise OSError("independent journal write failure")

    monkeypatch.setattr(ledger, "settle", fail_settlement)
    with pytest.raises(OSError):
        generate(provider)
    key = next(iter(ledger.amounts))
    saved = store.get("calls", key)
    assert saved and not saved["cost_is_upper_bound"]
    monkeypatch.setattr(ledger, "settle", settle)
    assert generate(provider) == (key, saved) and key in ledger.settled
    assert sum(r.url.path == "/v1/chat/completions" for r, _ in requests) == 1


def test_reservation_without_response_is_never_replayed(tmp_path, monkeypatch):
    provider, store, ledger, requests = setup_provider(tmp_path, monkeypatch)
    put = store.put

    def interrupted_response(namespace, key, value):
        if namespace == "calls":
            raise OSError("independent response write failure")
        put(namespace, key, value)

    monkeypatch.setattr(store, "put", interrupted_response)
    with pytest.raises(OSError):
        generate(provider)
    key = next(iter(ledger.amounts))
    assert key not in ledger.settled and store.get("calls", key) is None
    monkeypatch.setattr(store, "put", put)
    resumed_key, record = generate(provider)
    assert resumed_key == key and record["error"] == "uncertain_previous_request"
    assert record["cost_usd"] == pytest.approx(0.03) and record["cost_is_upper_bound"]
    assert sum(r.url.path == "/v1/chat/completions" for r, _ in requests) == 1


@pytest.mark.parametrize("override", [
    {"model_revision": "b" * 40}, {"temperature": 0.2}, {"seed": 10},
    {"vllm_version": "0.28.1"}, {"max_model_len": 131072},
])
def test_changed_provenance_never_reuses_generation(tmp_path, monkeypatch, override):
    provider, _, _, _ = setup_provider(tmp_path, monkeypatch)
    first_key, _ = generate(provider)
    changed, _, ledger, requests = setup_provider(
        tmp_path, monkeypatch, config=configuration(**override)
    )
    second_key, _ = generate(changed)
    assert first_key != second_key and len(ledger.amounts) == 2
    assert any(r.url.path == "/v1/chat/completions" for r, _ in requests)


def test_redirect_never_leaves_loopback_and_has_no_raw_error_retention(tmp_path, monkeypatch):
    def handler(request, _):
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                307, headers={"location": "https://example.com/exfil"},
                text="PRIVATE-HTTP-ERROR-CONTENT",
            )

    provider, store, ledger, requests = setup_provider(tmp_path, monkeypatch, handler=handler)
    key, record = generate(provider)
    assert record["status"] == "api_error" and record["http_status"] == 307
    assert key in ledger.settled and all(r.url.host == "127.0.0.1" for r, _ in requests)
    assert "PRIVATE-HTTP-ERROR-CONTENT" not in store.path("calls", key).read_text()


def test_returned_model_and_prompt_count_mismatch_fail_closed(tmp_path, monkeypatch):
    body = response_body()
    body["usage"] = {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17}
    provider, _, _, _ = setup_provider(tmp_path, monkeypatch, body=body)
    assert generate(provider)[1]["status"] == "invalid_output"
    body = response_body()
    body["model"] = "substituted-model"
    other, _, _, _ = setup_provider(tmp_path / "other", monkeypatch, body=body)
    assert generate(other)[1]["status"] == "invalid_output"


@pytest.mark.parametrize("finish_reason,status,cause", [
    ("stop", "ok", "accepted_text"),
    ("length", "invalid_output", "non_stop_finish_reason"),
    ("content_filter", "refusal", "content_filter"),
])
def test_finish_and_numeric_stop_metadata_survive_response_validation(
    tmp_path, monkeypatch, finish_reason, status, cause
):
    body = response_body()
    body["choices"][0].update(finish_reason=finish_reason, stop_reason=151645)
    provider, store, ledger, _ = setup_provider(tmp_path, monkeypatch, body=body)
    key, record = generate(provider)
    assert record["status"] == status
    assert record["diagnostic_cause"] == cause
    assert record["finish_reason"] == record["stop_reason"] == finish_reason
    assert record["provider_stop_reason"] == {"kind": "token_id", "token_id": 151645}
    assert record["usage"]["output_tokens"] == 7
    assert record["text"] == ("Observed result [E0003]." if status == "ok" else "")
    assert store.get("calls", key) == record and key in ledger.settled


@pytest.mark.parametrize("finish_reason,want_reason,cause", [
    ("stop", "stop", "empty_content"),
    ("length", "length", "non_stop_finish_reason"),
    (None, "missing", "non_stop_finish_reason"),
])
def test_output_at_token_cap_does_not_infer_a_missing_length_finish_reason(
    tmp_path, monkeypatch, finish_reason, want_reason, cause
):
    body = response_body()
    body["choices"][0]["finish_reason"] = finish_reason
    body["choices"][0]["message"]["content"] = " "
    body["usage"].update(completion_tokens=3200, total_tokens=3220)
    provider, _, _, _ = setup_provider(tmp_path, monkeypatch, body=body)
    _, record = generate(provider, max_tokens=3200)
    assert record["status"] == "invalid_output" and record["text"] == ""
    assert record["usage"]["output_tokens"] == 3200
    assert record["finish_reason"] == want_reason
    assert record["diagnostic_cause"] == cause


@pytest.mark.parametrize("stop_reason,want", [
    (None, {"kind": "none"}),
    ("PRIVATE-STOP-TEXT", {"kind": "string_redacted"}),
    (True, {"kind": "invalid"}),
    (-1, {"kind": "invalid"}),
    (2**63, {"kind": "invalid"}),
    ({"reasoning": "PRIVATE-STOP-TEXT"}, {"kind": "invalid"}),
])
def test_unsafe_finish_and_stop_metadata_are_never_retained(
    tmp_path, monkeypatch, stop_reason, want
):
    body = response_body()
    body["choices"][0].update(
        finish_reason={"reasoning": "PRIVATE-FINISH-TEXT"}, stop_reason=stop_reason,
    )
    provider, store, _, _ = setup_provider(tmp_path, monkeypatch, body=body)
    key, record = generate(provider)
    assert record["status"] == "invalid_output" and record["text"] == ""
    assert record["finish_reason"] == record["stop_reason"] == "unknown"
    assert record["provider_stop_reason"] == want
    assert record["diagnostic_cause"] == "non_stop_finish_reason"
    retained = store.path("calls", key).read_text()
    assert "PRIVATE-FINISH-TEXT" not in retained and "PRIVATE-STOP-TEXT" not in retained


@pytest.mark.parametrize("mutation,cause", [
    ({"model": "PRIVATE-UNEXPECTED-MODEL"}, "model_mismatch"),
    ({"usage": None}, "invalid_usage"),
    ({"usage": {"prompt_tokens": 20, "completion_tokens": True, "total_tokens": 21}},
     "invalid_usage_counts"),
    ({"usage": {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 999}},
     "usage_total_mismatch"),
])
def test_early_rejections_keep_safe_finish_metadata_without_invalid_usage(
    tmp_path, monkeypatch, mutation, cause
):
    body = response_body()
    body.update(mutation)
    body["choices"][0]["finish_reason"] = "length"
    provider, store, _, _ = setup_provider(tmp_path, monkeypatch, body=body)
    key, record = generate(provider)
    assert record["status"] == "invalid_output" and record["text"] == ""
    assert record["usage"] == {} and record["diagnostic_cause"] == cause
    assert record["finish_reason"] == "length"
    assert record["provider_stop_reason"] == {"kind": "missing"}
    assert "PRIVATE-UNEXPECTED-MODEL" not in store.path("calls", key).read_text()


def test_malformed_json_has_explicit_response_diagnostic(tmp_path, monkeypatch):
    def handler(request, _):
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, text="PRIVATE-MALFORMED-RESPONSE")

    provider, store, _, _ = setup_provider(tmp_path, monkeypatch, handler=handler)
    key, record = generate(provider)
    assert record["status"] == "invalid_output"
    assert record["diagnostic_cause"] == "malformed_response"
    assert record["finish_reason"] == record["stop_reason"] == "missing"
    assert record["provider_stop_reason"] == {"kind": "missing"}
    assert "PRIVATE-MALFORMED-RESPONSE" not in store.path("calls", key).read_text()
