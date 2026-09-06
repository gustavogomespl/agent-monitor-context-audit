# Qwen Colab implementation plan

> For agentic workers: use the existing research plan and this implementation
> contract. Work in the current `pilot` checkout; do not commit, publish, or run
> paid generation during implementation.

**Goal:** Run the unchanged four-condition experiment using one pinned
`Qwen/Qwen3.8-27B` on a Colab H100, with a reproducible, opt-in notebook.

**Architecture:** A short-lived vLLM server bound to loopback supplies text-only
generation and exact tokenizer counts. The existing runner selects one backend
per run; Anthropic remains available for compatibility. A Colab supervisor owns
the server and experiment subprocesses, enforces a session time budget, and
records GPU session costs and persistent results. CPU tests mock the HTTP and
process boundaries, never a missing real dataset.

**Scientific scope:** The user's September 6 request authorizes the infrastructure
revision described in the preceding review. The hypothesis, prompts, paired
split, four conditions, token budget rule, failure escalation and freeze gates
remain the research-plan baseline. No test scores have been generated.

## Tasks

- [x] Add a loopback-only Qwen provider with pinned model/engine provenance,
  identical counting/generation templates, thinking disabled, bounded requests,
  durable cache, explicit runtime-based request cost and synthetic tests.
- [x] Integrate backend configuration and CLI selection; preserve completed rows
  on resume, reject incomplete CLI success, and account for GPU session overhead.
- [x] Build a managed Colab launch helper and empty-output notebook covering
  installation, persistent workspace, pins, data acquisition, pilot/development,
  explicit freeze/test, export and CPU analysis. No default live generation.
- [x] Fix private-path validation and frozen-bootstrap reproduction with regressions.
- [x] Update English methodology, decisions, setup documentation and dependency
  metadata; run the complete CPU suite, Ruff and independent final review.

## Shared implementation contract

`AuditConfig.provider` is `anthropic` (legacy default) or `qwen_local`.
`AuditConfig.qwen` holds a `QwenConfig` with `model_revision`, `vllm_version`,
`base_url` (loopback), `max_model_len`, `gpu_hourly_rate_usd`, generation parameters
and a generation seed. Model and tokenizer share the exact immutable revision.
The Qwen backend requires the same model in both roles. Config templates leave
revision, hourly rate and data/rubric approval unset until Colab setup.

`QwenProvider(store, ledger, config, *, timeout, max_calls, canaries)` implements
`model_info`, `count_text`, `count_request`, and `generate` with the same result
shape as the Anthropic provider. `ledger` is the existing request BudgetLedger;
Qwen reserves timeout times hourly rate, settles observed request duration and
retains uncertain reservations. Cache keys include all model, engine, sampling,
template and repeat/attempt identities. HTTP clients disable ambient proxies and
redirects. No tool definitions or transcript command execution.

The Colab supervisor separately measures and bounds the whole managed GPU session,
including engine startup, counts, failed calls, teardown and retries. It persists
session reservations before process launch and never treats an interrupted unknown
session as free. Request costs and session overhead are reported separately; this
is a cost estimate at the user's supplied effective hourly rate, not a Colab bill.
GPU allocation before/after the managed block remains explicitly outside its cap;
the notebook offers automatic runtime disconnection after durable completion.

Committed notebooks remain inert and output-free. Model installation and data
setup are explicit steps; generation additionally requires live opt-in, an exact
model revision, a positive rate/cap and data-use/rubric confirmation. The same
private run directory resumes cache identities across Colab sessions.

## Verification

Write focused synthetic regression tests before implementing each behavior.
Run `uv run pytest` and `uv run ruff check .` with environment-only sandbox
workarounds if necessary. Inspect notebook source and execute its default offline
path. Actual H100/VRAM/performance validation remains a real pilot deliverable.

Implementation validation on September 6: 169 synthetic CPU tests passed, Ruff
passed, and all three notebooks executed in their default offline mode with
committed outputs empty. Independent review included actual synthetic process
cleanup and launch-registration checks. No model or GPU was started.
