"""Bounded loopback Qwen inference, exact counts and durable GPU-time estimates.

The managed Colab supervisor pins the weights and enforces the session deadline.
HTTP timeouts bound individual I/O phases, not the lifetime of a GPU allocation.
"""

from __future__ import annotations

import json
import math
import re
import time
from urllib.parse import urlsplit, urlunsplit

import httpx

from context_audit.provider import BudgetLedger, TokenCountError, Usage, utc_now
from context_audit.runtime_models import QwenConfig
from context_audit.storage import PrivateStore, digest

MODEL = "Qwen/Qwen3.8-27B"
TEMPLATE = {"enable_thinking": False, "preserve_thinking": False}


def _integer(value, *, minimum=0) -> bool:
    return type(value) is int and value >= minimum


def _response_metadata(data) -> dict:
    """Bounded diagnostics only: unknown strings and stop sequences may contain data."""
    choices = data.get("choices") if isinstance(data, dict) else None
    choice = (
        choices[0] if isinstance(choices, list) and len(choices) == 1
        and isinstance(choices[0], dict) else {}
    )
    reason = choice.get("finish_reason")
    if reason is None:
        finish_reason = "missing"
    elif isinstance(reason, str) and reason in (
        "stop", "length", "content_filter", "tool_calls", "function_call", "error", "abort",
    ):
        finish_reason = reason
    else:
        finish_reason = "unknown"
    stop = choice.get("stop_reason")
    if "stop_reason" not in choice:
        provider_stop_reason = {"kind": "missing"}
    elif stop is None:
        provider_stop_reason = {"kind": "none"}
    elif _integer(stop) and stop <= 2**31 - 1:
        provider_stop_reason = {"kind": "token_id", "token_id": stop}
    elif isinstance(stop, str):
        provider_stop_reason = {"kind": "string_redacted"}
    else:
        provider_stop_reason = {"kind": "invalid"}
    return {
        "finish_reason": finish_reason,
        # Preserve the existing generic stop_reason alias for the finish reason.
        "stop_reason": finish_reason,
        "provider_stop_reason": provider_stop_reason,
    }


class QwenProvider:
    """One independent text request at a time; no keys, tools or reasoning retention."""

    def __init__(
        self,
        store: PrivateStore,
        ledger: BudgetLedger,
        config: QwenConfig,
        *,
        timeout: float = 60,
        max_calls: int = 1000,
        canaries: list[str] | None = None,
    ):
        self.config = QwenConfig.model_validate(config.model_dump())
        self.config.validate_live()
        if ledger.unit == "usd" and self.config.gpu_hourly_rate_usd is None:
            raise ValueError("Dollar accounting requires an explicit GPU hourly rate")
        if not math.isfinite(timeout) or timeout <= 0 or not _integer(max_calls, minimum=1):
            raise ValueError("Positive finite timeout and maximum generation calls required")
        self.store, self.ledger = store, ledger
        self.timeout, self.max_calls = timeout, max_calls
        self.canaries = list(canaries or [])
        self.version = self.config.vllm_version
        self.provenance = {
            "provider": "qwen_local", "config": self.config.model_dump(),
            "template": TEMPLATE, "provider_contract": "qwen-local-v1",
        }
        if ledger.unit == "seconds":
            self.provenance["accounting_unit"] = "seconds"
        self.counter_method = (
            f"vllm.tokenize/exact-body-and-chat/v1; vllm={self.version}; "
            f"model={MODEL}; revision={self.config.model_revision}; "
            "enable_thinking=false; preserve_thinking=false"
        )
        # Avoid DNS resolution even for the accepted localhost alias.
        url = urlsplit(self.config.base_url)
        base_url = self.config.base_url
        if url.hostname == "localhost":
            base_url = urlunsplit(("http", f"127.0.0.1:{url.port}", "", "", ""))
        self.client = httpx.Client(
            base_url=base_url, timeout=timeout, trust_env=False, follow_redirects=False,
        )
        self._info: dict | None = None

    @staticmethod
    def _require_model(model: str) -> None:
        if model != MODEL:
            raise ValueError("The Qwen provider requires the exact Qwen/Qwen3.8-27B model")

    def model_info(self, model: str) -> dict:
        self._require_model(model)
        if self._info is not None:
            return dict(self._info)
        try:
            version = self.client.get("/version")
            version.raise_for_status()
            if version.json().get("version") != self.version:
                raise ValueError("version mismatch")
            response = self.client.get("/v1/models")
            response.raise_for_status()
            models = response.json().get("data")
            if not isinstance(models, list) or len(models) != 1:
                raise ValueError("model inventory mismatch")
            info = models[0]
            if (
                not isinstance(info, dict) or info.get("id") != MODEL
                or not _integer(info.get("max_model_len"), minimum=1)
                or info["max_model_len"] != self.config.max_model_len
            ):
                raise ValueError("model or context mismatch")
        except (httpx.HTTPError, ValueError, AttributeError, TypeError):
            raise ValueError("Qwen server model, version or context validation failed") from None
        self._info = {
            "id": MODEL, "max_input_tokens": self.config.max_model_len,
            "max_tokens": self.config.max_model_len,
            "model_revision": self.config.model_revision, "vllm_version": self.version,
            "revision_verification": "pinned_supervised_launch; not_attested_by_models_endpoint",
            "counter_method": self.counter_method,
        }
        self.store.put("models", digest(self.provenance), {
            "at": utc_now(), "requested": model, "info": self._info,
        })
        return dict(self._info)

    @staticmethod
    def _chat(model: str, system: str, text: str) -> dict:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": text})
        return {
            "model": model, "messages": messages, "add_generation_prompt": True,
            "add_special_tokens": False, "chat_template_kwargs": dict(TEMPLATE),
        }

    def _count(self, payload: dict) -> int:
        key = digest({"request": payload, "provenance": self.provenance})
        try:
            self.model_info(payload["model"])
            cached = self.store.get("token_counts", key)
            if cached is not None:
                if not _integer(cached.get("tokens")):
                    raise ValueError("Invalid cached token count")
                return cached["tokens"]
            response = self.client.post("/tokenize", json=payload)
            response.raise_for_status()
            data = response.json()
            tokens, ids = data.get("count"), data.get("tokens")
            if (
                not _integer(tokens) or not isinstance(ids, list) or len(ids) != tokens
                or any(not _integer(token) for token in ids)
                or not _integer(data.get("max_model_len"), minimum=1)
                or data["max_model_len"] != self.config.max_model_len
            ):
                raise ValueError("Invalid token count response")
        except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
            self.store.append("errors", {
                "phase": "token_count", "request_hash": key,
                "error_type": type(exc).__name__, "at": utc_now(), "canaries": self.canaries,
            })
            raise TokenCountError("Qwen token counting failed; no count was inferred") from None
        self.store.put("token_counts", key, {
            "tokens": tokens, "method": self.counter_method,
            "at": utc_now(), "canaries": self.canaries,
        })
        return tokens

    def count_text(self, model: str, text: str) -> int:
        self._require_model(model)
        if not text:
            return 0
        return self._count({"model": model, "prompt": text, "add_special_tokens": False})

    def count_request(self, model: str, system: str, text: str) -> int:
        self._require_model(model)
        return self._count(self._chat(model, system, text))

    def _record(self, status: str, **fields) -> dict:
        return {
            "status": status, "text": "", "model": MODEL, "usage": {},
            **self._accounting(0), "cost_is_upper_bound": False, "latency_seconds": 0,
            "cost_method": (
                "generation_elapsed_seconds_times_supplied_gpu_hourly_rate"
                if self.config.gpu_hourly_rate_usd is not None
                else "unpriced_gpu_time; no_USD_cost_inferred"
            ),
            "at": utc_now(), "canaries": self.canaries, **fields,
        }

    def _accounting(self, seconds: float) -> dict:
        rate = self.config.gpu_hourly_rate_usd
        cost = seconds * rate / 3600 if rate is not None else None
        return {
            "cost_usd": cost, "gpu_seconds": seconds,
            "accounting_unit": self.ledger.unit,
            "accounting_amount": seconds if self.ledger.unit == "seconds" else cost,
        }

    @staticmethod
    def _response(data: dict, *, count: int, max_tokens: int) -> dict:
        # Capture safe termination metadata before any validation can fail. A token
        # count equal to max_tokens alone never establishes a length termination.
        metadata = _response_metadata(data)

        def invalid(cause: str, usage: dict | None = None) -> dict:
            return {
                "status": "invalid_output", "text": "", "usage": usage or {},
                "diagnostic_cause": cause, **metadata,
            }

        if not isinstance(data, dict):
            return invalid("invalid_response_type")
        if data.get("model") != MODEL:
            return invalid("model_mismatch")
        usage = data.get("usage")
        if not isinstance(usage, dict):
            return invalid("invalid_usage")
        if any(not _integer(usage.get(k)) for k in (
            "prompt_tokens", "completion_tokens", "total_tokens"
        )):
            return invalid("invalid_usage_counts")
        if usage["prompt_tokens"] != count:
            return invalid("prompt_token_mismatch")
        if usage["completion_tokens"] > max_tokens:
            return invalid("completion_token_limit_exceeded")
        if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
            return invalid("usage_total_mismatch")
        details = usage.get("completion_tokens_details")
        if details is not None and (
            not isinstance(details, dict)
            or not _integer(details.get("reasoning_tokens", 0))
        ):
            return invalid("invalid_reasoning_usage")
        if details is not None and details.get("reasoning_tokens", 0) != 0:
            return invalid("reasoning_tokens_present")
        cached_tokens = 0
        prompt_details = usage.get("prompt_tokens_details")
        if prompt_details is not None:
            if not isinstance(prompt_details, dict):
                return invalid("invalid_prompt_token_details")
            cached_tokens = prompt_details.get("cached_tokens", 0)
            if not _integer(cached_tokens) or cached_tokens > count:
                return invalid("invalid_cached_token_count")
        clean_usage = Usage(
            input_tokens=count, output_tokens=usage["completion_tokens"],
            cache_read_input_tokens=cached_tokens,
        ).model_dump()
        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            return invalid("invalid_choices")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            return invalid("invalid_message")
        if message.get("role") != "assistant":
            return invalid("invalid_message_role")
        reason = choice.get("finish_reason")
        if message.get("refusal") or reason == "content_filter":
            return {
                "status": "refusal", "text": "", "usage": clean_usage, **metadata,
                "diagnostic_cause": (
                    "refusal_present" if message.get("refusal") else "content_filter"
                ),
            }
        content = message.get("content")
        if reason != "stop":
            return invalid("non_stop_finish_reason", clean_usage)
        if message.get("reasoning") or message.get("reasoning_content") or message.get("thinking"):
            return invalid("reasoning_content_present", clean_usage)
        if message.get("tool_calls") or message.get("function_call"):
            return invalid("tool_call_present", clean_usage)
        if not isinstance(content, str):
            return invalid("non_text_content", clean_usage)
        if not content.strip():
            return invalid("empty_content", clean_usage)
        if re.search(r"<\s*/?\s*think(?:ing)?\b", content, re.IGNORECASE):
            return invalid("thinking_tag_present", clean_usage)
        return {
            "status": "ok", "text": content, "usage": clean_usage,
            "diagnostic_cause": "accepted_text", **metadata,
        }

    def generate(
        self, *, model: str, system: str, text: str, max_tokens: int,
        context_window: int, identity: dict, structured_schema: dict | None = None,
    ) -> tuple[str, dict]:
        self._require_model(model)
        if not _integer(max_tokens, minimum=1) or not _integer(context_window, minimum=1):
            raise ValueError("Positive integer output and context limits required")
        seed = int(digest({"seed": self.config.seed, "identity": identity})[:8], 16) % (2**31)
        request = {
            **self._chat(model, system, text), "max_tokens": max_tokens, "stream": False,
            "n": 1, "seed": seed, "temperature": self.config.temperature,
            "top_p": self.config.top_p, "top_k": self.config.top_k,
            "presence_penalty": self.config.presence_penalty,
        }
        if structured_schema is not None:
            # Decoder constraints are part of the durable request/cache identity. They do
            # not alter the tokenized chat or introduce tool/function calls.
            request["structured_outputs"] = {
                "json": json.loads(json.dumps(structured_schema, allow_nan=False)),
            }
        payload = {
            "request": request, "context_window": context_window,
            "identity": identity, "provenance": self.provenance,
        }
        key = digest(payload)
        cached = self.store.get("calls", key)
        if cached is not None:
            if (
                key in self.ledger.amounts and key not in self.ledger.settled
                and not cached.get("cost_is_upper_bound", True)
            ):
                self.ledger.settle(key, cached.get("accounting_amount", cached["cost_usd"]))
            return key, cached
        if key in self.ledger.amounts:
            reserved = self.ledger.amounts[key]
            seconds = (
                reserved if self.ledger.unit == "seconds"
                else reserved * 3600 / self.config.gpu_hourly_rate_usd
            )
            return key, self._record(
                "api_error", **self._accounting(seconds), cost_is_upper_bound=True,
                error="uncertain_previous_request",
            )
        if len(self.ledger.amounts) >= self.max_calls:
            return key, self._record("budget_violation", error="maximum_generation_calls")
        try:
            count = self.count_request(model, system, text)
        except TokenCountError:
            record = self._record("api_error", error="token_count_failed")
            self.store.put("calls", key, record)
            return key, record
        if count + max_tokens > min(context_window, self.config.max_model_len):
            return key, self._record("context_limit", estimated_input_tokens=count)
        reservation = self._accounting(self.timeout)
        try:
            self.ledger.reserve(key, reservation["accounting_amount"])
        except ValueError:
            error = "gpu_time_cap" if self.ledger.unit == "seconds" else "financial_cap"
            return key, self._record("budget_violation", error=error)
        self.store.put("requests", key, {**payload, "canaries": self.canaries, "at": utc_now()})
        started = time.monotonic()
        record = self._record("api_error", **reservation, cost_is_upper_bound=True)
        try:
            response = self.client.post("/v1/chat/completions", json=request)
        except httpx.HTTPError as exc:
            record["error_type"] = type(exc).__name__
            record["latency_seconds"] = time.monotonic() - started
            self.store.put("calls", key, record)
            return key, record
        elapsed = time.monotonic() - started
        record.update(
            latency_seconds=elapsed, **self._accounting(elapsed),
            cost_is_upper_bound=False,
        )
        if response.is_success:
            try:
                record.update(self._response(response.json(), count=count, max_tokens=max_tokens))
            except (ValueError, TypeError, AttributeError):
                record.update(
                    status="invalid_output", text="", error="malformed_response",
                    diagnostic_cause="malformed_response", **_response_metadata(None),
                )
        else:
            record.update(status="api_error", error="http_error", http_status=response.status_code)
        # Crash between save and settlement recovers the same response without generation.
        self.store.put("calls", key, record)
        self.ledger.settle(key, record["accounting_amount"])
        return key, record
