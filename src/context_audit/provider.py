"""One real provider, bounded calls, explicit prices and durable cost reservations."""

from __future__ import annotations

import importlib.metadata
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pydantic import Field

from context_audit.runtime_models import StrictModel
from context_audit.storage import PrivateStore, digest


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TokenCountError(ValueError):
    """Safe count-endpoint failure; never convert an unavailable count into zero."""


class Usage(StrictModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_read_input_tokens: int = Field(default=0, ge=0)
    cache_creation_input_tokens: int = Field(default=0, ge=0)


class Price(StrictModel):
    input_per_million: float = Field(gt=0)
    output_per_million: float = Field(gt=0)
    cache_read_per_million: float = Field(ge=0)
    cache_write_per_million: float = Field(ge=0)

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.input_per_million
            + usage.output_tokens * self.output_per_million
            + usage.cache_read_input_tokens * self.cache_read_per_million
            + usage.cache_creation_input_tokens * self.cache_write_per_million
        ) / 1_000_000

    def reserve(self, context_window: int, max_output: int) -> float:
        # Reserve a context-window upper bound, not an optimistic estimated prompt size.
        rate = max(
            self.input_per_million, self.cache_read_per_million, self.cache_write_per_million
        )
        return (context_window * rate + max_output * self.output_per_million) / 1_000_000


def load_prices(path: Path, models: list[str]) -> tuple[dict[str, Price], dict]:
    if not path.exists():
        raise ValueError("Price snapshot missing; fill configs/prices.example.yaml locally")
    data = yaml.safe_load(path.read_text())
    if (
        not isinstance(data, dict)
        or data.get("currency") != "USD"
        or data.get("unit") != "per_million_tokens"
    ):
        raise ValueError("Prices require USD per_million_tokens units")
    if not data.get("verified_at") or not data.get("source_url", "").startswith(
        ("https://platform.claude.com/", "https://www.anthropic.com/")
    ):
        raise ValueError("Prices require verification date and official source URL")
    verified = datetime.fromisoformat(str(data["verified_at"]))
    if verified.date() > datetime.now(timezone.utc).date():
        raise ValueError("Price verification date cannot be in the future")
    try:
        prices = {model: Price.model_validate(data["models"][model]) for model in models}
    except (KeyError, TypeError) as exc:
        raise ValueError("Provide exact model IDs with verified prices") from exc
    return prices, data


class BudgetLedger:
    """Each request is reserved before network I/O. Uncertain charges stay reserved."""

    def __init__(self, store: PrivateStore, limit: float):
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("Explicit positive finite dollar budget required")
        self.store, self.limit = store, limit
        self.amounts: dict[str, float] = {}
        self.settled: set[str] = set()
        for entry in store.read_journal("budget"):
            key = entry["call_id"]
            if entry["action"] == "reserve":
                if key in self.amounts:
                    raise ValueError("Duplicate budget reservation")
                self.amounts[key] = entry["amount"]
            elif entry["action"] == "settle" and key in self.amounts:
                self.amounts[key] = entry["amount"]
                self.settled.add(key)
            else:
                raise ValueError("Invalid budget journal")
        if self.committed > limit + 1e-12:
            raise ValueError("Existing spend/reservations exceed requested budget")

    @property
    def committed(self) -> float:
        return sum(self.amounts.values())

    def reserve(self, call_id: str, amount: float) -> None:
        if call_id in self.amounts:
            raise ValueError("Call already reserved; reconcile uncertain charge")
        if amount < 0 or not math.isfinite(amount) or self.committed + amount > self.limit:
            raise ValueError("Insufficient remaining budget for a conservative request reservation")
        self.store.append(
            "budget", dict(action="reserve", call_id=call_id, amount=amount, at=utc_now())
        )
        self.amounts[call_id] = amount

    def settle(self, call_id: str, actual: float) -> None:
        if call_id not in self.amounts or actual < 0 or not math.isfinite(actual):
            raise ValueError("Invalid settled charge")
        self.store.append(
            "budget", dict(action="settle", call_id=call_id, amount=actual, at=utc_now())
        )
        self.amounts[call_id] = actual
        self.settled.add(call_id)
        if self.committed > self.limit + 1e-12:
            raise ValueError(
                "Actual provider usage exceeded reservation; stop and reconcile pricing"
            )


class AnthropicProvider:
    """No tools, shared conversation, hidden thinking request, temperature or seed."""

    def __init__(
        self,
        store: PrivateStore,
        ledger: BudgetLedger,
        prices: dict[str, Price],
        *,
        timeout: float = 60,
        max_calls: int = 1000,
        canaries: list[str] | None = None,
    ):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY is missing; real execution cannot use fixtures")
        import anthropic

        self.client = anthropic.Anthropic(max_retries=0, timeout=timeout)
        self.store, self.ledger, self.prices = store, ledger, prices
        self.max_calls, self.canaries = max_calls, canaries or []
        self.version = importlib.metadata.version("anthropic")
        self.counter_method = f"anthropic.messages.count_tokens/differential/v1; sdk={self.version}"

    def model_info(self, model: str) -> dict:
        info = self.client.models.retrieve(model).model_dump(mode="json")
        self.store.put("models", digest(model), dict(at=utc_now(), requested=model, info=info))
        return info

    def count_request(self, model: str, system: str, text: str) -> int:
        key = digest(dict(model=model, system=system, text=text, sdk=self.version))
        cached = self.store.get("token_counts", key)
        if cached is not None:
            return cached["tokens"]
        args = dict(model=model, messages=[{"role": "user", "content": text or " "}])
        if system:
            args["system"] = system
        import anthropic

        try:
            tokens = self.client.messages.count_tokens(**args).input_tokens
        except anthropic.APIError as exc:
            self.store.append(
                "errors",
                dict(
                    phase="token_count",
                    request_hash=key,
                    error_type=type(exc).__name__,
                    at=utc_now(),
                    canaries=self.canaries,
                ),
            )
            raise TokenCountError("Provider token counting failed; no count was inferred") from None
        self.store.put(
            "token_counts",
            key,
            dict(tokens=tokens, method=self.counter_method, at=utc_now(), canaries=self.canaries),
        )
        return tokens

    def count_text(self, model: str, text: str) -> int:
        if not text:
            return 0
        # Removes one-message overhead approximately. Counts are provider estimates.
        baseline = self.count_request(model, "", " ")
        return max(1, self.count_request(model, "", text) - baseline)

    def generate(
        self,
        *,
        model: str,
        system: str,
        text: str,
        max_tokens: int,
        context_window: int,
        identity: dict,
    ) -> tuple[str, dict]:
        payload = dict(
            model=model,
            system=system,
            text=text,
            max_tokens=max_tokens,
            context_window=context_window,
            identity=identity,
            sdk=self.version,
        )
        key = digest(payload)
        cached = self.store.get("calls", key)
        if cached is not None:
            if (
                key in self.ledger.amounts
                and key not in self.ledger.settled
                and not cached.get("cost_is_upper_bound", True)
            ):
                self.ledger.settle(key, cached["cost_usd"])
            return key, cached
        if key in self.ledger.amounts:
            # A crash after sending may have been billed. Never send it again automatically.
            return key, dict(
                status="api_error",
                text="",
                model=model,
                cost_usd=self.ledger.amounts[key],
                cost_is_upper_bound=True,
                latency_seconds=0,
                usage={},
                error="uncertain_previous_request",
            )
        if len(self.ledger.amounts) >= self.max_calls:
            return key, dict(
                status="budget_violation",
                text="",
                cost_usd=0,
                latency_seconds=0,
                usage={},
                error="maximum_generation_calls",
            )
        try:
            count = self.count_request(model, system, text)
        except TokenCountError:
            record = dict(
                status="api_error",
                text="",
                model=model,
                usage={},
                cost_usd=0,
                cost_is_upper_bound=False,
                latency_seconds=0,
                error="token_count_failed",
                at=utc_now(),
                canaries=self.canaries,
            )
            self.store.put("calls", key, record)
            return key, record
        if count + max_tokens > context_window:
            return key, dict(
                status="context_limit",
                text="",
                cost_usd=0,
                latency_seconds=0,
                usage={},
                estimated_input_tokens=count,
            )
        reserve = self.prices[model].reserve(context_window, max_tokens)
        try:
            self.ledger.reserve(key, reserve)
        except ValueError:
            return key, dict(
                status="budget_violation",
                text="",
                cost_usd=0,
                latency_seconds=0,
                usage={},
                error="financial_cap",
            )
        # Request persistence carries source opt-out notices outside model-visible text.
        self.store.put("requests", key, dict(**payload, canaries=self.canaries, at=utc_now()))
        started = time.monotonic()
        record = dict(
            status="api_error",
            text="",
            usage={},
            model=model,
            at=utc_now(),
            cost_usd=reserve,
            cost_is_upper_bound=True,
            canaries=self.canaries,
        )
        try:
            message = self.client.messages.create(
                model=model,
                system=system,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": text}],
            )
            raw = message.model_dump(mode="json")
            raw["content"] = [b for b in raw.get("content", []) if b.get("type") == "text"]
            usage = Usage(**{k: getattr(message.usage, k, 0) or 0 for k in Usage.model_fields})
            cost = self.prices[model].cost(usage)
            reason = message.stop_reason
            status = {
                "refusal": "refusal",
                "model_context_window_exceeded": "context_limit",
                "max_tokens": "invalid_output",
            }.get(reason, "ok")
            if reason not in (
                "end_turn",
                "stop_sequence",
                "refusal",
                "model_context_window_exceeded",
                "max_tokens",
            ):
                status = "invalid_output"
            record.update(
                status=status,
                text="\n".join(block.text for block in message.content if block.type == "text"),
                usage=usage.model_dump(),
                raw_response=raw,
                model=message.model,
                stop_reason=reason,
                cost_usd=cost,
                cost_is_upper_bound=False,
            )
            # Save the response before clearing its reservation; recovery is conservative.
            record["latency_seconds"] = time.monotonic() - started
            self.store.put("calls", key, record)
            self.ledger.settle(key, cost)
        except Exception as exc:
            # Never store exception strings, which may echo request bodies/keys, publicly.
            record["error_type"] = type(exc).__name__
            record["status"] = "api_error"
            record["text"] = ""
        record["latency_seconds"] = time.monotonic() - started
        self.store.put("calls", key, record)
        return key, record
