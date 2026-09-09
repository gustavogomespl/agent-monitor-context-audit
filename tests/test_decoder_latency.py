"""Independent synthetic, clock-controlled checks for the CPU latency gate."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def latency():
    from context_audit import decoder_latency

    return decoder_latency


class ToyTokenizer:
    def __len__(self):
        return 248044

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(char) for char in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


def fake_runtime(
    monkeypatch, latency, *, mask_seconds=0.001, reject_token=False, mask_prefixes=None,
):
    clock = [0.0]
    operations = {"masks": 0, "tokens": 0}
    scripted = iter(mask_seconds) if isinstance(mask_seconds, list) else None

    class Matcher:
        def __init__(self, compiled):
            self.tokens = []

        def accept_token(self, token):
            operations["tokens"] += 1
            self.tokens.append(token)
            return not reject_token

        def fill_next_token_bitmask(self, bitmask):
            operations["masks"] += 1
            if mask_prefixes is not None:
                mask_prefixes.append((self, list(self.tokens)))
            elapsed = next(scripted, 0.001) if scripted is not None else mask_seconds
            if isinstance(elapsed, Exception):
                raise elapsed
            clock[0] += elapsed
            return True

    monkeypatch.setattr(latency.time, "perf_counter", lambda: clock[0])
    xgr = SimpleNamespace(
        GrammarMatcher=Matcher,
        allocate_token_bitmask=lambda batch_size, vocab_size: object(),
    )
    compiler = SimpleNamespace(compile_json_schema=lambda schema: object())
    return xgr, compiler, ToyTokenizer(), operations, clock


def test_profiles_real_token_prefixes_and_bounded_mask_sequence(monkeypatch, latency):
    xgr, compiler, tokenizer, operations, _ = fake_runtime(monkeypatch, latency)
    result = latency._profile_schema(
        xgr, compiler, tokenizer, {"type": "object"},
        name="monitor", vocab_size=248320, started=0,
    )
    assert result["mask_calls"] == operations["masks"] == 280
    assert result["accepted_token_calls"] == operations["tokens"] > 256
    assert len(result["positions"]) == 8
    assert {case["position"] for case in result["positions"]} == {0, 32, 128, 256}
    assert {case["text_kind"] for case in result["positions"]} == {"ascii", "json_escapes"}
    assert all(case["sequential_steps"] == 32 for case in result["positions"])
    assert result["max_mask_seconds"] == pytest.approx(0.001)
    assert len(result["schema_sha256"]) == 64


def test_slow_first_mask_fails_before_further_iterations_without_sleep(monkeypatch, latency):
    xgr, compiler, tokenizer, operations, _ = fake_runtime(
        monkeypatch, latency, mask_seconds=0.251,
    )
    with pytest.raises(latency.DecoderLatencyError, match="mask latency") as caught:
        latency._profile_schema(
            xgr, compiler, tokenizer, {}, name="monitor", vocab_size=248320, started=0,
        )
    assert operations["masks"] == 1
    assert caught.value.receipt["reason_code"] == "mask_latency"
    assert caught.value.receipt["mask_seconds"] == 0.251
    assert caught.value.receipt["mask_gate_seconds"] == 0.25
    assert caught.value.receipt["profile"] == "monitor"
    assert caught.value.receipt["position"] == 0


def test_v7_transient_spike_requires_two_fast_same_state_masks_and_records_all_calls(
    monkeypatch, latency,
):
    prefixes = []
    xgr, compiler, tokenizer, operations, _ = fake_runtime(
        monkeypatch, latency, mask_seconds=[0.437, 0.145, 0.080],
        mask_prefixes=prefixes,
    )
    result = latency._profile_schema(
        xgr, compiler, tokenizer, {}, name="summary", vocab_size=248320,
        started=0, confirm_spikes=True,
    )
    assert operations["masks"] == result["mask_calls"] == 282
    assert len({id(item[0]) for item in prefixes[:3]}) == 3
    assert prefixes[0][1] == prefixes[1][1] == prefixes[2][1]
    assert result["max_mask_seconds"] == pytest.approx(0.437)
    assert result["max_gate_mask_seconds"] == pytest.approx(0.145)
    assert result["mask_gate_policy"] == "two_fast_confirmations_v1"
    first = result["positions"][0]
    assert result["confirmation_replay_token_calls"] == 2 * first["prefix_tokens"]
    assert result["accepted_token_calls"] == operations["tokens"]
    assert first["confirmation_replay_token_calls"] == 2 * first["prefix_tokens"]
    assert first["mask_seconds"][:3] == pytest.approx([0.437, 0.145, 0.080])
    assert len(first["mask_seconds"]) == 37
    assert first["latency_spikes"] == [{
        "initial_mask_index": 0,
        "initial_mask_seconds": pytest.approx(0.437),
        "confirmation_mask_seconds": pytest.approx([0.145, 0.080]),
        "confirmed": True,
    }]


@pytest.mark.parametrize("measurements,expected_calls", [
    ([0.437, 0.251], 2), ([0.437, 0.080, 0.251], 3), ([0.437, 0.437], 2),
])
def test_v7_any_slow_confirmation_fails_and_keeps_measured_times(
    monkeypatch, latency, measurements, expected_calls,
):
    xgr, compiler, tokenizer, operations, _ = fake_runtime(
        monkeypatch, latency, mask_seconds=measurements,
    )
    with pytest.raises(latency.DecoderLatencyError) as caught:
        latency._profile_schema(
            xgr, compiler, tokenizer, {}, name="summary", vocab_size=248320,
            started=0, confirm_spikes=True,
        )
    assert operations["masks"] == expected_calls
    assert caught.value.receipt["reason_code"] == "mask_latency"
    assert caught.value.receipt["mask_measurements_seconds"] == pytest.approx(measurements)
    spike = caught.value.receipt["latency_spikes"][0]
    assert not spike["confirmed"]
    assert spike["confirmation_mask_seconds"] == pytest.approx(measurements[1:])


def test_v7_confirmation_native_error_fails_without_leaking_exception_text(monkeypatch, latency):
    xgr, compiler, tokenizer, operations, _ = fake_runtime(
        monkeypatch, latency, mask_seconds=[0.437, RuntimeError("SYNTHETIC_PRIVATE_DETAIL")],
    )
    with pytest.raises(latency.DecoderLatencyError) as caught:
        latency._profile_schema(
            xgr, compiler, tokenizer, {}, name="summary", vocab_size=248320,
            started=0, confirm_spikes=True,
        )
    assert operations["masks"] == 2
    assert caught.value.receipt["reason_code"] == "native_mask_failed"
    assert caught.value.receipt["exception_type"] == "RuntimeError"
    assert "SYNTHETIC_PRIVATE_DETAIL" not in json.dumps(caught.value.receipt)


def test_v7_confirmation_deadline_still_fails(monkeypatch, latency):
    xgr, compiler, tokenizer, operations, clock = fake_runtime(
        monkeypatch, latency, mask_seconds=[0.251, 0.200, 0.100],
    )
    clock[0] = 119.6
    with pytest.raises(latency.DecoderLatencyError) as caught:
        latency._profile_schema(
            xgr, compiler, tokenizer, {}, name="summary", vocab_size=248320,
            started=0, confirm_spikes=True,
        )
    assert operations["masks"] == 2
    assert caught.value.receipt["reason_code"] == "total_deadline"
    assert caught.value.receipt["mask_measurements_seconds"] == pytest.approx([0.251, 0.200])


def test_v7_fresh_replay_includes_tokens_accepted_after_the_initial_prefix(monkeypatch, latency):
    prefixes = []
    xgr, compiler, tokenizer, operations, _ = fake_runtime(
        monkeypatch, latency, mask_seconds=[0.001] * 5 + [0.437, 0.080, 0.090],
        mask_prefixes=prefixes,
    )
    result = latency._profile_schema(
        xgr, compiler, tokenizer, {}, name="summary", vocab_size=248320,
        started=0, confirm_spikes=True,
    )
    assert len({id(item[0]) for item in prefixes[5:8]}) == 3
    assert prefixes[5][1] == prefixes[6][1] == prefixes[7][1] == prefixes[0][1] + [ord("a")] * 2
    assert result["confirmation_replay_token_calls"] == 2 * len(prefixes[5][1])
    assert result["accepted_token_calls"] == operations["tokens"]


def test_v7_failing_later_case_preserves_earlier_case_mask_measurements(monkeypatch, latency):
    xgr, compiler, tokenizer, operations, _ = fake_runtime(
        monkeypatch, latency, mask_seconds=[0.001] * 35 + [0.437, 0.251],
    )
    with pytest.raises(latency.DecoderLatencyError) as caught:
        latency._profile_schema(
            xgr, compiler, tokenizer, {}, name="summary", vocab_size=248320,
            started=0, confirm_spikes=True,
        )
    receipt = caught.value.receipt
    assert receipt["mask_calls"] == operations["masks"] == 37
    assert len(receipt["positions"]) == 2
    assert receipt["positions"][0]["mask_seconds"] == pytest.approx([0.001] * 35)
    assert receipt["positions"][1]["mask_seconds"] == pytest.approx([0.437, 0.251])
    assert receipt["accepted_token_calls"] == operations["tokens"]


def test_v7_confirmation_prefix_replay_failure_is_fatal(monkeypatch, latency):
    xgr, compiler, tokenizer, operations, _ = fake_runtime(
        monkeypatch, latency, mask_seconds=[0.437],
    )
    original = xgr.GrammarMatcher
    instances = []

    def matcher(compiled):
        instance = original(compiled)
        instances.append(instance)
        if len(instances) == 2:
            instance.accept_token = lambda token: False
        return instance

    xgr.GrammarMatcher = matcher
    with pytest.raises(latency.DecoderLatencyError) as caught:
        latency._profile_schema(
            xgr, compiler, tokenizer, {}, name="summary", vocab_size=248320,
            started=0, confirm_spikes=True,
        )
    assert operations["masks"] == 1
    assert caught.value.receipt["reason_code"] == "confirmation_replay_failed"


def test_v7_deadline_after_sequential_token_preserves_all_completed_masks(monkeypatch, latency):
    xgr, compiler, tokenizer, operations, clock = fake_runtime(monkeypatch, latency)
    original = xgr.GrammarMatcher

    def matcher(compiled):
        instance = original(compiled)
        accept = instance.accept_token

        def consume(token):
            result = accept(token)
            if operations["masks"] == 4:
                clock[0] = 121
            return result

        instance.accept_token = consume
        return instance

    xgr.GrammarMatcher = matcher
    with pytest.raises(latency.DecoderLatencyError) as caught:
        latency._profile_schema(
            xgr, compiler, tokenizer, {}, name="summary", vocab_size=248320,
            started=0, confirm_spikes=True,
        )
    receipt = caught.value.receipt
    assert receipt["reason_code"] == "total_deadline"
    assert receipt["mask_calls"] == operations["masks"] == 4
    assert receipt["positions"][0]["mask_seconds"] == pytest.approx([0.001] * 4)
    assert receipt["accepted_token_calls"] == operations["tokens"]


def test_total_deadline_fails_before_compiling_or_masking(monkeypatch, latency):
    xgr, compiler, tokenizer, operations, clock = fake_runtime(monkeypatch, latency)
    clock[0] = 121
    with pytest.raises(TimeoutError, match="deadline"):
        latency._profile_schema(
            xgr, compiler, tokenizer, {}, name="monitor", vocab_size=248320, started=0,
        )
    assert operations == {"tokens": 0, "masks": 0}


def test_invalid_tokenized_prefix_fails_before_masking(monkeypatch, latency):
    xgr, compiler, tokenizer, operations, _ = fake_runtime(
        monkeypatch, latency, reject_token=True,
    )
    with pytest.raises(ValueError, match="prefix"):
        latency._profile_schema(
            xgr, compiler, tokenizer, {}, name="monitor", vocab_size=248320, started=0,
        )
    assert operations == {"tokens": 1, "masks": 0}


@pytest.mark.parametrize("model,revision", [
    ("other/model", "a" * 40), ("Qwen/Qwen3.8-27B", "main"),
    ("Qwen/Qwen3.8-27B", "a" * 39), ("Qwen/Qwen3.8-27B", "z" * 40),
])
def test_invalid_pin_fails_before_loading_tokenizer(monkeypatch, latency, model, revision):
    monkeypatch.setattr(latency, "_load_runtime", lambda *args: pytest.fail("must not load"))
    with pytest.raises(ValueError, match="pin"):
        latency.check_decoder_latency(model, revision)


@pytest.mark.parametrize("config_style", ["nested", "plain", "none"])
def test_loader_uses_only_pinned_tokenizer_config_and_full_text_vocabulary(
    monkeypatch, latency, config_style,
):
    calls = []
    tokenizer = ToyTokenizer()

    def load_tokenizer(model, **kwargs):
        calls.append(("tokenizer", model, kwargs))
        return tokenizer

    def load_config(model, **kwargs):
        calls.append(("config", model, kwargs))
        config = SimpleNamespace(vocab_size=248320, _commit_hash="a" * 40)
        if config_style == "nested":
            config.text_config = SimpleNamespace(vocab_size=248320)
            config.vocab_size = 1  # The text vocabulary must take priority.
        elif config_style == "none":
            config.text_config = None
        return config

    def from_huggingface(loaded, *, vocab_size):
        assert loaded is tokenizer
        calls.append(("xgrammar", vocab_size))
        return SimpleNamespace(vocab_size=vocab_size)

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        AutoConfig=SimpleNamespace(from_pretrained=load_config),
    ))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda count: None))
    monkeypatch.setitem(sys.modules, "xgrammar", SimpleNamespace(
        TokenizerInfo=SimpleNamespace(from_huggingface=from_huggingface),
    ))
    monkeypatch.setattr(latency.importlib.metadata, "version", lambda name: "0.2.3")
    _, loaded_tokenizer, info, _, pin = latency._load_runtime("Qwen/Qwen3.8-27B", "a" * 40)
    assert loaded_tokenizer is tokenizer and info.vocab_size == 248320
    assert pin == {"config_revision_pin": "a" * 40, "loaded_config_revision": "a" * 40}
    expected = {"revision": "a" * 40, "trust_remote_code": False}
    assert calls == [
        ("tokenizer", "Qwen/Qwen3.8-27B", expected),
        ("config", "Qwen/Qwen3.8-27B", expected),
        ("xgrammar", 248320),
    ]


def test_wrong_xgrammar_version_fails_before_transformers_import(monkeypatch, latency):
    monkeypatch.setattr(latency.importlib.metadata, "version", lambda name: "0.3.0")
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ValueError, match="xgrammar==0.2.3"):
        latency._load_runtime("Qwen/Qwen3.8-27B", "a" * 40)


@pytest.mark.parametrize("vocab_size,tokenizer_size", [(95, 95), (248320, 248321), (True, 1)])
def test_full_vocabulary_validation_rejects_toy_or_inconsistent_sizes(
    latency, vocab_size, tokenizer_size,
):
    with pytest.raises(ValueError, match="vocabulary"):
        latency._validate_vocabulary(vocab_size, tokenizer_size)


def test_receipt_profiles_both_schemas_with_512_visible_ids(monkeypatch, latency):
    xgr, compiler, tokenizer, _, _ = fake_runtime(monkeypatch, latency)
    xgr.GrammarCompiler = lambda info, **kwargs: compiler
    monkeypatch.setattr(latency, "_load_runtime", lambda *args: (
        xgr, tokenizer, SimpleNamespace(vocab_size=248320), {"xgrammar": "0.2.3"},
        {"config_revision_pin": "a" * 40, "loaded_config_revision": None},
    ))
    receipt = latency.check_decoder_latency("Qwen/Qwen3.8-27B", "a" * 40)
    assert receipt["status"] == "passed" and receipt["model_generation_executed"] is False
    assert receipt["vocab_size"] == 248320
    assert receipt["visible_event_count"] == 512
    assert receipt["tokenizer_revision"] == "a" * 40
    assert receipt["config_revision_pin"] == "a" * 40
    assert receipt["loaded_config_revision"] is None
    assert receipt["max_mask_seconds"] == pytest.approx(0.001)
    assert receipt["max_gate_mask_seconds"] == pytest.approx(0.001)
    assert receipt["mask_gate_policy"] == "single_measurement_v1"
    assert {profile["name"] for profile in receipt["profiles"]} == {"summary", "monitor"}
    assert receipt["mask_calls"] == 560


def test_v7_latency_receipt_binds_the_selected_production_schema(monkeypatch, latency):
    xgr, compiler, tokenizer, _, _ = fake_runtime(monkeypatch, latency)
    compiled_schemas = []
    compiler.compile_json_schema = lambda schema: compiled_schemas.append(schema) or object()
    xgr.GrammarCompiler = lambda info, **kwargs: compiler
    monkeypatch.setattr(latency, "_load_runtime", lambda *args: (
        xgr, tokenizer, SimpleNamespace(vocab_size=248320), {"xgrammar": "0.2.3"},
        {"config_revision_pin": "a" * 40, "loaded_config_revision": None},
    ))
    receipt = latency.check_decoder_latency(
        "Qwen/Qwen3.8-27B", "a" * 40,
        structured_summary_mode="schema_citations_separate_ids_v1",
    )
    assert receipt["structured_summary_mode"] == "schema_citations_separate_ids_v1"
    assert receipt["status"] == "passed"
    assert receipt["mask_calls"] == 581
    assert receipt["mask_gate_policy"] == "two_fast_confirmations_v1"
    assert {profile["mask_gate_policy"] for profile in receipt["profiles"]} == {
        "two_fast_confirmations_v1"
    }
    assert receipt["confirmation_replay_token_calls"] == 0
    prefixes = [
        case for case in receipt["profiles"][0]["positions"]
        if case["text_kind"].startswith("event_prefix_")
    ]
    assert len(prefixes) == 7
    assert all(case["position"] == 0 and case["sequential_steps"] == 0 for case in prefixes)
    assert compiled_schemas[0]["properties"]["environment_and_state"]["items"][
        "properties"
    ]["text"]["pattern"]


@pytest.mark.parametrize("initial_fast_masks", [0, 301])
def test_v7_receipt_confirms_spikes_for_summary_and_monitor(
    monkeypatch, latency, initial_fast_masks,
):
    xgr, compiler, tokenizer, operations, _ = fake_runtime(
        monkeypatch, latency, mask_seconds=[0.001] * initial_fast_masks + [0.437, 0.080, 0.090],
    )
    xgr.GrammarCompiler = lambda info, **kwargs: compiler
    monkeypatch.setattr(latency, "_load_runtime", lambda *args: (
        xgr, tokenizer, SimpleNamespace(vocab_size=248320), {"xgrammar": "0.2.3"},
        {"config_revision_pin": "a" * 40, "loaded_config_revision": None},
    ))
    result = latency.check_decoder_latency(
        "Qwen/Qwen3.8-27B", "a" * 40,
        structured_summary_mode="schema_citations_separate_ids_v1",
    )
    assert result["mask_calls"] == operations["masks"] == 583
    assert result["max_mask_seconds"] == pytest.approx(0.437)
    assert result["max_gate_mask_seconds"] == pytest.approx(0.090)
    assert result["accepted_token_calls"] == operations["tokens"]
    assert result["confirmation_replay_token_calls"] > 0


def test_v7_monitor_failure_keeps_completed_summary_profile_and_actual_counts(monkeypatch, latency):
    xgr, compiler, tokenizer, operations, _ = fake_runtime(
        monkeypatch, latency, mask_seconds=[0.001] * 301 + [0.437, 0.251],
    )
    xgr.GrammarCompiler = lambda info, **kwargs: compiler
    monkeypatch.setattr(latency, "_load_runtime", lambda *args: (
        xgr, tokenizer, SimpleNamespace(vocab_size=248320), {"xgrammar": "0.2.3"},
        {"config_revision_pin": "a" * 40, "loaded_config_revision": None},
    ))
    with pytest.raises(latency.DecoderLatencyError) as caught:
        latency.check_decoder_latency(
            "Qwen/Qwen3.8-27B", "a" * 40,
            structured_summary_mode="schema_citations_separate_ids_v1",
        )
    receipt = caught.value.receipt
    assert receipt["mask_calls"] == operations["masks"] == 303
    assert receipt["accepted_token_calls"] == operations["tokens"]
    assert receipt["profile"] == "monitor"
    assert len(receipt["completed_profiles"]) == 1
    summary = receipt["completed_profiles"][0]
    assert summary["name"] == "summary" and summary["mask_calls"] == 301
    assert sum(len(case["mask_seconds"]) for case in summary["positions"]) == 301
    assert receipt["positions"][0]["mask_seconds"] == pytest.approx([0.437, 0.251])


def test_v7_final_deadline_keeps_both_completed_profiles(monkeypatch, latency):
    xgr, compiler, tokenizer, operations, clock = fake_runtime(monkeypatch, latency)
    xgr.GrammarCompiler = lambda info, **kwargs: compiler
    monkeypatch.setattr(latency, "_load_runtime", lambda *args: (
        xgr, tokenizer, SimpleNamespace(vocab_size=248320), {"xgrammar": "0.2.3"},
        {"config_revision_pin": "a" * 40, "loaded_config_revision": None},
    ))
    original = latency._profile_schema

    def profile(*args, **kwargs):
        receipt = original(*args, **kwargs)
        if kwargs["name"] == "monitor":
            clock[0] = 121
        return receipt

    monkeypatch.setattr(latency, "_profile_schema", profile)
    with pytest.raises(latency.DecoderLatencyError) as caught:
        latency.check_decoder_latency(
            "Qwen/Qwen3.8-27B", "a" * 40,
            structured_summary_mode="schema_citations_separate_ids_v1",
        )
    receipt = caught.value.receipt
    assert receipt["reason_code"] == "total_deadline"
    assert receipt["mask_calls"] == operations["masks"] == 581
    assert receipt["accepted_token_calls"] == operations["tokens"]
    assert len(receipt["completed_profiles"]) == 2


def test_latency_cli_forwards_v7_mode_to_the_isolated_worker(monkeypatch, latency, capsys):
    def run(command, **kwargs):
        index = command.index("--structured-summary-mode")
        assert command[index + 1] == "schema_citations_separate_ids_v1"
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            "status": "passed", "model_generation_executed": False,
            "structured_summary_mode": "schema_citations_separate_ids_v1",
        }))

    monkeypatch.setattr(latency.subprocess, "run", run)
    assert latency.main([
        "--model", "Qwen/Qwen3.8-27B", "--revision", "a" * 40,
        "--structured-summary-mode", "schema_citations_separate_ids_v1",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"


def test_cli_hard_timeout_returns_failed_json_and_nonzero(monkeypatch, latency, capsys):
    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 150
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(latency.subprocess, "run", timeout)
    result = latency.main(["--model", "Qwen/Qwen3.8-27B", "--revision", "a" * 40])
    receipt = json.loads(capsys.readouterr().out)
    assert result == 1
    assert receipt["status"] == "failed" and receipt["reason_code"] == "hard_timeout"
    assert receipt["model_generation_executed"] is False


def test_cli_never_accepts_a_failed_worker_as_passed(monkeypatch, latency, capsys):
    failed = {
        "status": "failed", "reason_code": "mask_latency", "model_generation_executed": False,
        "phase": "mask", "exception_type": "DecoderLatencyError",
        "mask_seconds": 0.251, "mask_gate_seconds": 0.25,
    }
    monkeypatch.setattr(latency.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=1, stdout=json.dumps(failed), stderr="",
    ))
    result = latency.main(["--model", "Qwen/Qwen3.8-27B", "--revision", "a" * 40])
    assert result == 1
    assert json.loads(capsys.readouterr().out) == failed


@pytest.mark.parametrize("stage", ["tokenizer", "config"])
def test_loading_failure_records_category_and_type_without_exception_text(
    monkeypatch, latency, stage, capsys,
):
    def load_tokenizer(*args, **kwargs):
        if stage == "tokenizer":
            raise OSError("SYNTHETIC_SECRET_DO_NOT_REPORT")
        return ToyTokenizer()

    def load_config(*args, **kwargs):
        raise OSError("SYNTHETIC_SECRET_DO_NOT_REPORT")

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        AutoConfig=SimpleNamespace(from_pretrained=load_config),
    ))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda count: None))
    monkeypatch.setitem(sys.modules, "xgrammar", SimpleNamespace())
    monkeypatch.setattr(latency.importlib.metadata, "version", lambda name: "0.2.3")
    result = latency.main([
        "--worker", "--model", "Qwen/Qwen3.8-27B", "--revision", "a" * 40,
    ])
    output = capsys.readouterr().out
    receipt = json.loads(output)
    assert result == 1 and receipt["reason_code"] == f"{stage}_load_failed"
    assert receipt["exception_type"] == "OSError"
    assert "SYNTHETIC_SECRET" not in output


def test_mismatched_loaded_config_revision_fails_before_grammar_setup(monkeypatch, latency):
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: ToyTokenizer()),
        AutoConfig=SimpleNamespace(from_pretrained=lambda *args, **kwargs: SimpleNamespace(
            vocab_size=248320, _commit_hash="b" * 40,
        )),
    ))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda count: None))
    monkeypatch.setitem(sys.modules, "xgrammar", SimpleNamespace())
    monkeypatch.setattr(latency.importlib.metadata, "version", lambda name: "0.2.3")
    with pytest.raises(latency.DecoderLatencyError) as caught:
        latency._load_runtime("Qwen/Qwen3.8-27B", "a" * 40)
    assert caught.value.receipt["reason_code"] == "config_revision_mismatch"
