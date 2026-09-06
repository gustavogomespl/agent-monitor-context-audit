# Agent Monitor Context Audit implementation plan

**Goal:** Implement and verify the user's complete retrospective study protocol,
then run paid development/test experiments only when access and budget are supplied.
**Architecture:** Typed label-free transcript inputs flow through four representations
to an independent Anthropic monitor. Evaluator-only labels join saved numeric outputs
for paired analysis. All content-bearing artifacts remain private.
**Stack:** Python 3.11+, Pydantic, Anthropic SDK, NumPy/pandas/scikit-learn,
matplotlib, pytest, nbformat/nbclient; uv lockfile; no agent framework or server.
**Authority:** `research_plan.md` (user-provided and authorized for implementation).

- [x] M1 acquisition/inventory component: verify official format/access/references, acquire pinned encrypted source,
  decrypt with official procedure, inventory sessions/families, opaque split manifests.
- [x] M2: schemas/parser/render, token budgets/head-tail, private atomic persistence,
  independent fixtures, invariant tests and two executable offline notebooks.
- [x] M3 infrastructure: provider, equal-priority prompts, retries, strict cost cap,
  token counting, context preflight, per-call cache/usage and pilot CLI.
- [ ] M3 empirical: 3–5 development pairs after model/access/budget confirmation.
- [ ] M4: human review of hypothesis/rubric/pilot; freeze protocol with source hashes
  and protocol-v1 commit/tag before any test scoring.
- [ ] M5: complete test run under explicit budget, incremental resumable accounting.
- [ ] M6: paired bootstrap, coverage/failures/cost/figures, qualitative worksheet,
  report, sanitized outputs; publication waits for explicit permission.

## Integration contracts and checks

Dataset exports `TranscriptInput` and separate `EvaluationLabel` objects. Runner
exports rows: transcript_id, scenario_id, family_id, split, label, condition,
repetition, data_origin, status, suspicion_score, escalate, body_tokens,
representation_tokens, input_tokens, output_tokens, cache_read_input_tokens,
cache_creation_input_tokens, summary_cost_usd, monitor_cost_usd, cost_usd,
cost_is_upper_bound, latency_seconds, monitor_model, summarizer_model. No text, paths or source names.
Analysis rejects synthetic origin for empirical reporting and requires complete
four-condition pairs for primary AUROC; group bootstrap uses family_id.

Each implementation unit gets focused tests, then integrated full-suite verification.
Final audit compares actual artifacts against all research-plan requirements.
Live blockers never become fabricated empirical success.

## Current evidence / decisions

2026-09-05: Initial repository has only Apache-2.0 LICENSE, commit 70e39fd.
No Anthropic API key is set in the shell. No paid calls authorized or made.
The supplied architectural design is the approved implementation scope; an additional
brainstorming approval loop would repeat the user's existing authorization.
Implementation is local in the supplied workspace; no branch change, commit or push yet.

## Verified checkpoint

2026-09-05: Source and packaged runtime checks pass 84 tests including both
notebook executions. Ruff and git diff whitespace checks pass. Synthetic demo
uses zero generation calls; empty-results analysis reports experiment_not_executed.
Publication scan checks 63 candidate files against 175 protected files, with zero
findings, including a regression for short canary identifiers.

Actual normalized data: 35 pairs, 33 families, 8 development and 27 test pairs,
zero cross-split families. API key remains absent, model IDs and prices unset,
zero real generation records. Official model/context preflight and human family
review keep M1 as a whole partial; M3 empirical through M6 remain pending.
See completion-audit.md for the requirement-by-requirement full-scope audit.
No new Git commit/tag/push, publication, or paid request was performed.

2026-09-06 continuation: corrected notebook live-entry cwd/.env handling and
explicit token-counter failure propagation without losing prior summary cost.
Installed-package validation now passes 91 tests on both Python 3.11.4 and 3.12.12;
Ruff/content scan pass (65 candidate files, 175 protected files). The existing live
access/model/budget/review blocker remains; no generation or publication occurred.
