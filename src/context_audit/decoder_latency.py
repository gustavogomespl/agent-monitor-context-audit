"""Bounded CPU mask profiling with the pinned Qwen tokenizer, never model weights."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import time

from context_audit.monitor_schema import monitor_response_schema
from context_audit.structured_summary import structured_summary_schema

MODEL = "Qwen/Qwen3.8-27B"
XGRAMMAR_VERSION = "0.2.3"
MAX_MASK_SECONDS = 0.25
MAX_PROFILE_SECONDS = 120
HARD_TIMEOUT_SECONDS = 150
POSITIONS = (0, 32, 128, 256)
SEQUENTIAL_STEPS = 32
SUMMARY_MODES = ("schema_citations_compact_v1", "schema_citations_separate_ids_v1")


class DecoderLatencyError(ValueError):
    """A safe categorical failure containing only explicitly selected diagnostics."""

    def __init__(self, reason_code: str, *, phase: str, **details):
        super().__init__(reason_code.replace("_", " "))
        self.receipt = _failure(reason_code, phase=phase, **details)


def _exception_type(exc: Exception) -> str:
    name = type(exc).__name__
    return name if re.fullmatch(r"[A-Za-z0-9_]{1,64}", name) else "Exception"


def _validate_pin(model: str, revision: str) -> None:
    if (
        model != MODEL or not isinstance(revision, str)
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
    ):
        raise DecoderLatencyError("invalid_model_pin", phase="configuration")


def _validate_vocabulary(vocab_size: int, tokenizer_size: int) -> None:
    if (
        type(vocab_size) is not int or vocab_size < 200000
        or type(tokenizer_size) is not int or not 200000 <= tokenizer_size <= vocab_size
    ):
        raise DecoderLatencyError("invalid_vocabulary", phase="tokenizer_config")


def _load_runtime(model: str, revision: str):
    version = importlib.metadata.version("xgrammar")
    if version != XGRAMMAR_VERSION:
        raise ValueError(f"Latency check requires xgrammar=={XGRAMMAR_VERSION}")
    # No AutoModel or model checkpoint is loaded. XGrammar's masks remain on CPU.
    try:
        import torch
        import xgrammar as xgr
        from transformers import AutoConfig, AutoTokenizer

        torch.set_num_threads(1)
    except Exception as exc:
        raise DecoderLatencyError(
            "runtime_import_failed", phase="runtime_import", exception_type=_exception_type(exc),
        ) from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
    except Exception as exc:
        raise DecoderLatencyError(
            "tokenizer_load_failed", phase="tokenizer_load", exception_type=_exception_type(exc),
        ) from exc
    try:
        config = AutoConfig.from_pretrained(model, revision=revision, trust_remote_code=False)
    except Exception as exc:
        raise DecoderLatencyError(
            "config_load_failed", phase="config_load", exception_type=_exception_type(exc),
        ) from exc
    loaded_revision = getattr(config, "_commit_hash", None)
    if loaded_revision is not None and loaded_revision != revision:
        raise DecoderLatencyError("config_revision_mismatch", phase="config_load")
    text_config = getattr(config, "text_config", None) or config
    vocab_size = getattr(text_config, "vocab_size", None)
    _validate_vocabulary(vocab_size, len(tokenizer))
    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    if info.vocab_size != vocab_size:
        raise DecoderLatencyError("xgrammar_vocabulary_mismatch", phase="tokenizer_config")
    versions = {
        package: importlib.metadata.version(package)
        for package in ("xgrammar", "torch", "transformers", "tokenizers")
    }
    return xgr, tokenizer, info, versions, {
        "config_revision_pin": revision, "loaded_config_revision": loaded_revision,
    }


def _check_deadline(started: float) -> None:
    if time.perf_counter() - started > MAX_PROFILE_SECONDS:
        raise TimeoutError("Decoder latency diagnostic exceeded its total deadline")


def _measure_mask(matcher, bitmask, started: float, *, profile: str, position: int) -> float:
    _check_deadline(started)
    before = time.perf_counter()
    try:
        matcher.fill_next_token_bitmask(bitmask)
    except Exception as exc:
        raise DecoderLatencyError(
            "native_mask_failed", phase="mask", profile=profile, position=position,
            exception_type=_exception_type(exc),
        ) from exc
    elapsed = time.perf_counter() - before
    if elapsed > MAX_MASK_SECONDS:
        raise DecoderLatencyError(
            "mask_latency", phase="mask", profile=profile, position=position,
            mask_seconds=elapsed, mask_gate_seconds=MAX_MASK_SECONDS,
        )
    _check_deadline(started)
    return elapsed


def _profile_schema(xgr, compiler, tokenizer, schema: dict, *, name: str,
                    vocab_size: int, started: float, extra_suffixes: tuple = ()) -> dict:
    """Measure full-vocabulary masks after real token ingestion at bounded positions."""
    _check_deadline(started)
    before = time.perf_counter()
    try:
        compiled = compiler.compile_json_schema(schema)
    except Exception as exc:
        raise DecoderLatencyError(
            "grammar_compile_failed", phase="grammar_compile", profile=name,
            exception_type=_exception_type(exc),
        ) from exc
    compile_seconds = time.perf_counter() - before
    _check_deadline(started)
    bitmask = xgr.allocate_token_bitmask(1, vocab_size)
    prefixes = {
        "summary": '{"environment_and_state":[{"text":"',
        "monitor": '{"suspicion_score":25.5,"brief_justification":"',
    }
    prefix = prefixes[name]
    escaped = json.dumps(
        'Toy "quote", backslash \\, newline\nand Unicode café 😀.', ensure_ascii=False,
    )[1:-1]
    single = tokenizer.encode("a", add_special_tokens=False)
    if len(single) != 1 or tokenizer.decode(single) != "a":
        raise ValueError("Tokenizer cannot supply the expected synthetic sequential token")
    results, total_tokens = [], 0
    for position in POSITIONS:
        extras = extra_suffixes if position == 0 else ()
        for kind, suffix in (("ascii", ""), ("json_escapes", escaped), *extras):
            matcher = xgr.GrammarMatcher(compiled)
            text = prefix + "a" * position + suffix
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if not 1 <= len(tokens) <= 4096 or tokenizer.decode(
                tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False,
            ) != text:
                raise ValueError("Synthetic prefix tokenization is not exact and bounded")
            accept_seconds = 0.0
            for token in tokens:
                _check_deadline(started)
                before = time.perf_counter()
                if not matcher.accept_token(token):
                    raise ValueError("Decoder rejected a valid synthetic token prefix")
                accept_seconds += time.perf_counter() - before
            total_tokens += len(tokens)
            # Repeated masks and subsequent changing states exercise both paths.
            durations = [
                _measure_mask(matcher, bitmask, started, profile=name, position=position)
                for _ in range(3)
            ]
            steps = 0 if (kind, suffix) in extras else SEQUENTIAL_STEPS
            for _ in range(steps):
                durations.append(_measure_mask(
                    matcher, bitmask, started, profile=name, position=position,
                ))
                before = time.perf_counter()
                if not matcher.accept_token(single[0]):
                    raise ValueError("Decoder rejected a valid synthetic sequential token")
                accept_seconds += time.perf_counter() - before
                total_tokens += 1
                _check_deadline(started)
            results.append({
                "position": position, "text_kind": kind, "prefix_tokens": len(tokens),
                "sequential_steps": steps, "mask_seconds": durations,
                "accept_token_seconds": accept_seconds,
            })
    timings = [duration for result in results for duration in result["mask_seconds"]]
    return {
        "name": name,
        "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
        "compile_seconds": compile_seconds, "positions": results,
        "mask_calls": len(timings), "accepted_token_calls": total_tokens,
        "max_mask_seconds": max(timings), "total_mask_seconds": sum(timings),
    }


def check_decoder_latency(
    model: str, revision: str, *, structured_summary_mode: str = "schema_citations_compact_v1",
) -> dict:
    """Profile only independent synthetic data with the immutable public tokenizer pin."""
    _validate_pin(model, revision)
    if structured_summary_mode not in SUMMARY_MODES:
        raise DecoderLatencyError("invalid_summary_mode", phase="configuration")
    started = time.perf_counter()
    xgr, tokenizer, info, versions, pin = _load_runtime(model, revision)
    _check_deadline(started)
    compiler = xgr.GrammarCompiler(info, max_threads=8, cache_enabled=False)
    visible_ids = {f"E{number:04d}" for number in range(1, 513)}
    schemas = {
        "summary": structured_summary_schema(
            visible_ids, token_budget=2048, text_bounds=False,
            citations_in_text=structured_summary_mode != "schema_citations_separate_ids_v1",
        ),
        "monitor": monitor_response_schema(visible_ids),
    }
    profiles = [
        _profile_schema(
            xgr, compiler, tokenizer, schema, name=name,
            vocab_size=info.vocab_size, started=started,
            extra_suffixes=(
                tuple((f"event_prefix_{i}", " " + suffix) for i, suffix in enumerate((
                    "E", "E0", "E00", "E000", "E0000", "caféE0000", "E٠٠٠٠",
                )))
                if name == "summary"
                and structured_summary_mode == "schema_citations_separate_ids_v1"
                else ()
            ),
        )
        for name, schema in schemas.items()
    ]
    _check_deadline(started)
    return {
        "status": "passed", "model_generation_executed": False,
        "model": model, "tokenizer_revision": revision,
        "structured_summary_mode": structured_summary_mode,
        **pin,
        "vocab_size": info.vocab_size, "tokenizer_vocab_size": len(tokenizer),
        "visible_event_count": len(visible_ids), "runtime_versions": versions,
        "machine": platform.machine(), "python_version": platform.python_version(),
        "max_mask_seconds": max(profile["max_mask_seconds"] for profile in profiles),
        "mask_gate_seconds": MAX_MASK_SECONDS, "total_gate_seconds": MAX_PROFILE_SECONDS,
        "hard_timeout_seconds": HARD_TIMEOUT_SECONDS,
        "elapsed_seconds": time.perf_counter() - started,
        "mask_calls": sum(profile["mask_calls"] for profile in profiles),
        "accepted_token_calls": sum(profile["accepted_token_calls"] for profile in profiles),
        "profiles": profiles,
    }


def _failure(reason_code: str, *, phase: str = "diagnostic",
             exception_type: str = "DecoderLatencyError", **details) -> dict:
    return {
        "status": "failed", "reason_code": reason_code, "model_generation_executed": False,
        "phase": phase, "exception_type": exception_type, **details,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--structured-summary-mode", choices=SUMMARY_MODES, default=SUMMARY_MODES[0],
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        _validate_pin(args.model, args.revision)
        if args.worker:
            receipt = check_decoder_latency(
                args.model, args.revision, structured_summary_mode=args.structured_summary_mode,
            )
        else:
            # A subprocess timeout can kill a stuck native compiler/mask operation;
            # an in-process elapsed-time check cannot interrupt a blocked C++ call.
            worker = subprocess.run(
                [sys.executable, "-m", "context_audit.decoder_latency", "--worker",
                 "--model", args.model, "--revision", args.revision,
                 "--structured-summary-mode", args.structured_summary_mode],
                capture_output=True, text=True, timeout=HARD_TIMEOUT_SECONDS,
            )
            try:
                receipt = json.loads(worker.stdout)
            except (json.JSONDecodeError, TypeError):
                receipt = _failure("invalid_worker_output", phase="worker")
            if not isinstance(receipt, dict) or receipt.get("status") not in {"passed", "failed"}:
                receipt = _failure("invalid_worker_output", phase="worker")
            if worker.returncode and receipt.get("status") == "passed":
                receipt = _failure("worker_failed", phase="worker")
    except DecoderLatencyError as exc:
        receipt = exc.receipt
    except subprocess.TimeoutExpired as exc:
        receipt = _failure("hard_timeout", phase="worker", exception_type=_exception_type(exc))
    except TimeoutError as exc:
        receipt = _failure("total_deadline", exception_type=_exception_type(exc))
    except Exception as exc:
        receipt = _failure("decoder_check_failed", exception_type=_exception_type(exc))
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
