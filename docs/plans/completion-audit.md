# Completion audit against the supplied research plan

**Audit date: 2026-09-05. Overall status: project implementation and real-data
inventory are available; the scientific study is not complete.** The scope is
the entire user-supplied [research_plan.md](../../research_plan.md), including
its milestones, artifacts and twelve quality criteria. Missing empirical
evidence is not replaced with a fixture run or a smaller definition of success.

This audit uses the current worktree and locally verified artifacts. An
implemented gate is not evidence that the gate has been passed. Test coverage
supports engineering contracts only; it does not validate the monitor's scientific
performance, the selected model's context capacity or human approval.

## Milestone audit

| Milestone / acceptance requirement | Current evidence | Status and remaining evidence |
| --- | --- | --- |
| M1: sources, official format/access, real pin, inventory, sessions/families, split and context limits | [References](../references.md) verifies R1–R12. Official source pin is `218c58315cc01ff0dc5a100e906c27d82d259521`. [Inventory](../../data/manifests/inventory.json) and [split](../../data/manifests/split.csv) exist. Strict local loading verifies 70 inputs and 70 separate labels; 35 pairs, 33 families, 8 development/27 test pairs, zero cross-split families. [Dataset evidence](dataset-evidence.md) records source/decryption/integrity checks and whole-case exclusions. | **Partially complete.** Acquisition and inventory are implemented and observed. Human family review, explicit model IDs, official model token counts and full/summarizer context-window acceptance remain pending. Byte sizes are not token counts. |
| M2: schemas/parser/render, full/head-tail, persistence, independent fixtures, label-leak tests, CPU notebooks | All named modules, required fixture tests and both notebooks exist. [Dataset evidence](dataset-evidence.md) reports 25 dataset/split/leakage tests and synthetic integration evidence. Current direct checks load the actual data, verify zero model-input canary matches and confirm empty outputs in both notebooks. Demo is explicitly synthetic with hand-authored summaries and no empirical score generation. | **Implemented and engineering-verified.** Latest full-suite results are recorded below. Synthetic behavior does not fulfill M3. |
| M3: real 3–5-pair development pilot with validated models/counts, real four-condition outputs, measured lengths/failures/cost | Official Anthropic adapter, shared-priority prompts, generation schema, retry/cost ledger and opt-in CLI are present. Pilot config selects three development pairs but has empty model IDs, null context windows and false review/data-use flags. There are no real provider request/call records or run/completion manifests. | **Not executed.** Requires API access, chosen validated models, verified dated prices, explicit stage budget, account data-use confirmation and live opt-in; then reviewed real outputs. |
| M4: complete development, freeze methods/split/models/prompts/threshold/failure policy at protocol-v1 | Freeze CLI checks reviewed, successful full-development evidence and signatures; the test runner checks matching frozen content and `protocol-v1` tag. Current Git HEAD remains the initial commit `70e39fd`; no protocol-v1 tag or `data/manifests/protocol-v1.json` exists. | **Not frozen.** Complete the pilot and full eight-pair development stage, review hypothesis/rubric/grouping/examples, prepare the freeze and create the reviewed commit/tag. No test-informed tuning has occurred because no test scores exist. |
| M5: main test with randomized order, cache, explicit cap, incremental records and complete accounting for missing cases | Runner and synthetic integration coverage provide the machinery. Main config uses the test split, all test pairs and one repetition. Real run/request/response/completion counts are zero. | **Not executed.** Requires M4 and authorized test budget. Need records for every planned unit or precise missing-unit reasons; a successful synthetic run is not this evidence. |
| M6: reproduce real results offline, review errors, figures/report, sanitize and publish only with authorization | [Results schema](../../results/public_scores.csv) is header-only; [metrics](../../results/metrics.json) has status `experiment_not_executed` and null primary outcome. Offline analysis/report/figure functions, notebook and private qualitative workflow exist. [Authored findings](../../reports/findings.md) separates inventory from unavailable performance. | **Analysis machinery implemented; empirical analysis/review/publication pending.** No empirical figures or conclusions exist. Obtain real scores, verify planned coverage, review errors and conclusions, scan content and obtain explicit publication authorization. |

## Twelve numbered quality criteria

| Plan criterion | Concrete current evidence | Verification scope / limit |
| --- | --- | --- |
| 1. Official parser separates canaries, human text and tool results | `dataset.parse_transcript`, `schemas.py`, `render.py`; dataset tests cover single-block lines, actual initial user text, malformed visible records, nested thinking, tool links, cwd and error flags. Actual 70-input load succeeds with zero source-canary matches in model objects. | **Implemented and observed on eligible real records plus independent fixtures.** The two malformed pairs are explicitly excluded, not silently repaired. Source notices remain private envelopes. |
| 2. Changing label/metadata/directory cannot change model prompts | `test_no_label_leakage.py` rejects evaluator fields in model-input types and checks invariant rendered text under independent directory/sidecar changes. Prompt constructors consume `TranscriptInput`, not `EvaluationLabel`. | **Engineering contract tested.** This prevents direct evaluator-field leakage; naturally informative trajectory evidence is still visible. |
| 3. Opaque IDs and no family crossing splits | Secret-key HMAC IDs in `dataset.py`; `test_splits.py`; opaque public split. Direct check finds 33 families and zero cross-split families across 70 labels. | **Observed structural invariant.** Automatic family identification still needs human semantic review; absence of a detected relationship does not prove independence. |
| 4. Full is not silently truncated; compressed views respect selected counter ceiling | `representations.py` preserves full, accounts for the omission marker and measures assembled head/tail; `test_budgets.py` covers ceilings/identity/malformed summaries. Runner performs full and summarizer context preflight and validates outputs with the monitor's counter. | **Implemented; fixture verified.** Real model-specific counts, API/context acceptance, summary lengths and counter discrepancy checks are pending the pilot. Offline demonstration byte estimates are explicitly not official token counts. |
| 5. Invalid JSON/refusal/missing score/API errors never become benign | `MonitorDecision` requires finite numeric scores; `MonitorResult` requires null failure scores and failure escalation. `test_failure_policy.py` and `test_runner.py` exercise invalid evidence, refusal and bounded repairs. | **Implemented with independent failure tests.** Real provider failure rates remain unknown. |
| 6. Costs include summaries/retries; offline analysis never invokes API | Provider disables implicit SDK retries, reserves each attempt before generation and records usage/cost. Runner aggregates both phases. Cost rows retain `cost_is_upper_bound`. Tests exercise all usage categories, retry accounting and restart reservations; reporting imports no API client for analysis. | **Engineering contract covered; empirical costs absent.** Reservations can exceed likely charges and caps are per run directory. Recorded latency sums generation calls, excluding token counting/preflight/local overhead. Unknown charges are not exact bills. |
| 7. Cache respects models, prompt versions and intentional repetitions | Generation cache payload includes model, system/request text, generation settings, SDK and an identity with normalized hash, config, condition, repetition, phase and attempt. `test_cache.py`, `test_runner.py` and independent integration evidence cover cache/resume; a direct crash-recovery probe settles a saved known charge without a new client call. | **Implemented with synthetic evidence.** Repetition is separate from retry. Frozen input/code/prompt integrity must remain valid for real resumption. |
| 8. Bootstrap preserves pairs, families and conditions; known metric fixtures | `metrics.py` common complete-pair set and family resampling; `test_metrics.py` covers known AUROC/ties/delta, family groups, repetitions, all-failure cases and planned missing pairs. `reporting.py` verifies sanitized planned-manifest identities. | **Implemented; independent metric evidence.** Real AUROC/intervals are absent. Observed rows alone cannot pass the planned-coverage quality gate; no low-FPR guarantee is claimed. |
| 9. Valid notebooks, no protected outputs without review | Both required notebooks are nbformat 4, each with five code cells, zero nonempty outputs and null execution counts. Notebook execution/format tests and `scripts/check_notebooks.py` exist. Publication tests include private content embedded in decoded notebook outputs. | **Current committed-form structure observed; offline execution separately validated by the integration checks.** Real notebook outputs remain private, and review precedes publication. |
| 10. Synthetic results never mixed into empirical results | Demo returns `synthetic_fixture`, zero generation calls and no empirical scores. Empirical reporting rejects fixture origins/extra textual columns. Public results contain no rows. | **Observed empty empirical state plus rejection tests.** Fixtures and authentic synthetic-benchmark records have distinct origins. |
| 11. Transcript data never executed; models receive no shell/tools | Parser/render/representations treat transcript commands as text. The provider calls only text Messages without tool definitions. Acquisition runs only reviewed Git/decryption operations; runner subprocess use is Git provenance/locking support, not transcript commands. | **Source audit and independent tests support the invariant.** Upstream verifiers/experiment runners are not invoked. No real generation occurred. |
| 12. No fabricated performance or exact-paper-replication claim | Header-only scores, null metrics, figure placeholders, authored findings and methodology explicitly state no experiment. Related work distinguishes visible-only monitoring from upstream configurations; summary direction is an untested hypothesis. | **Observed current artifacts.** Future positive, null or negative results need evidence and human review. |

## Named artifacts and operational checks

| Required artifacts / commands | Current evidence |
| --- | --- |
| Root files: README, LICENSE, pyproject.toml, uv.lock, .gitignore, .env.example, AGENTS.md, research_plan.md | All exist. Existing Apache-2.0 LICENSE is retained; upstream MIT notices are separately documented. The supplied Portuguese plan is preserved as the author-provided source; repository implementation/documentation is English. The lockfile resolves official `anthropic` 1.4.0 and no torch/CUDA/agent framework. |
| Configs: pilot.yaml, main.yaml, prices.example.yaml | All exist. Models/prices/review gates intentionally remain unset. Final dataset defaults point to `data/private`, matching the actual verified inventory. A full-development configuration is created from pilot by the documented copy/edit step, not silently treated as the three-pair pilot. |
| Prompts: monitor.txt, summary_common.txt, summary_free.txt, summary_structured.txt | All exist as generic templates. Common preservation priorities are shared; monitor policy/task/schema are condition-independent. No condition receives another condition's decision. |
| Source modules: __init__, cli, schemas, dataset, render, representations, provider, runner, metrics, reporting | All named modules exist, with small additional runtime_models, storage and sanitation modules. No web application, persistent server, training or agent framework was added. |
| Required tests: dataset, no_label_leakage, splits, budgets, failure_policy, metrics, cache; independent fixtures | All exist, plus CLI/runner/reporting/publication/notebook/integration coverage. CI does not acquire real benchmark data or require API keys. |
| Both named notebooks | Both exist and have empty outputs; they distinguish optional live pilot execution from API-free real-results reproduction. Notebook content/runtime audit is covered separately by the analysis worker and final integrated checks. |
| data/README.md, data/manifests/, data/private/, runs/private/ | All exist. Private paths are ignored; public inventory/split contain aggregate/opaque data. Actual private source and normalized inputs remain local. |
| results/public_scores.csv, metrics.json, figures/ | All exist as honest empty-results artifacts. `figures/README.md` lists the three future real-result figures; absence of fabricated PNGs is intentional, not empirical completion. `run_manifest.json` will be exported from a real run and does not exist yet. |
| Five named docs and reports/findings.md | All exist. Detailed authored findings were restored after an accidental placeholder overwrite; generated `results/findings.md` remains a separate machine report. AI assistance and pending author review are explicit. |
| .github/workflows/tests.yml | Exists on `ubuntu-latest`, explicitly syncs locked dependencies for Python 3.11, then runs pytest, Ruff, demo, empty-result analysis, notebook checks and publication scan. Only synthetic tests run; no acquisition command or live flag is present. No remote GitHub run has been performed in this session. |
| Python portability | Actual local Python 3.11.4 and 3.12.12 environments each pass all 91 tests on the latest revision. POSIX `fcntl` locking matches the target macOS and Ubuntu CI platforms. This is not Windows support or proof of a remote Ubuntu CI run. |
| Default CLI/docs | Help paths checked offline for acquire, inventory, run, freeze, export, analyze and scan-public. README includes the acquisition extra, full-development step, explicit stage budgets, review flag, export and planned result manifest. Its initial setup/demo is API-free. |
| Publication controls | `scan-public` scans actual candidate content and decoded notebook outputs, not only paths. Exact-text scanning cannot prove absence of paraphrases/raster leaks; source-less CI scans report their limited corpus coverage. Human content/history review and explicit publication permission remain necessary. |

## Verification record and limits

Direct scope-audit checks on 2026-09-05 established:

- Seventy real normalized inputs load with separate labels; there are 16
  development and 54 test transcripts, 33 families and no cross-split families.
- Both notebook source files have empty outputs and no execution counts.
- Public scores contain only their header; metrics report no experiment.
- No real generation request/call records, run manifests or completion manifests
  exist beneath `runs/private`; no paid generation has been performed.
- No protocol-v1 freeze artifact or tag exists. The worktree has no new commits
  or public push; scientific review remains outstanding.
- Every named plan artifact exists except artifacts that require an executed
  experiment (real scores, run provenance export, empirical figures and freeze).
- The current content scan passed across 63 public candidate files and 175
  protected source/derived files, with zero findings and the private corpus
  available. This scan is exact-text evidence, not a semantic leak guarantee.
  All local links in the ten inspected Markdown documents resolve.

The dataset worker's targeted checks and synthetic end-to-end evidence are
recorded in [dataset-evidence.md](dataset-evidence.md). Final integrated
`uv run pytest`, `uv run ruff check .`, demo, notebook and scan results should
be appended below after the last relevant edits. A prior green check must not be
used to certify subsequent code changes. This audit does not mark the thread
goal complete or imply that empirical milestones have been achieved.

### Final integrated verification, 2026-09-05

- `uv run --locked --no-editable pytest`: **84 passed in 5.67s**, Python 3.12.12.
  This includes both actual offline notebook kernel executions and a real provider
  implementation exercised through an isolated fake SDK boundary. It does not
  constitute real API inference. Local kernel sockets required approved execution
  outside the filesystem sandbox; tests force `RUN_LIVE=false`.
- `.venv/bin/ruff check .`: all checks passed, including notebook cells.
- `.venv/bin/context-audit demo`: four synthetic representations, zero generation
  calls, no empirical score export.
- `.venv/bin/context-audit analyze`: `experiment_not_executed`, null primary result.
- `.venv/bin/context-audit scan-public`: passed, 63 candidate files, 175 protected
  source files, protected corpus available, no findings. A separate regression
  proves that copied canary identifiers shorter than 80 characters are detected.
- `git diff --check`: passed. No new commit, tag or push.
- Fresh authenticated-data loader check: 70 inputs, 35 pairs, 33 families, 16
  development and 54 test transcripts, zero crossing families, retained canaries.
  API key absent in shell and local `.env`; real generation records: zero.

The non-editable package install avoids this macOS environment's hidden `.pth`
behavior, independently confirming that the package and CLI install successfully.
M2 and the infrastructure portions of M3/M6 are now engineering-verified.
The empirical study remains unexecuted; the goal is not marked complete.

### Continuation verification, 2026-09-06

The preceding goal turn made substantive implementation progress. This continuation
revalidated the actual pending live prerequisites and fixed two concrete entry-path
issues plus explicit count-failure handling; it was not a completed empirical run.

- Notebook 01 now loads the root `.env` only after explicit live opt-in, preserves
  an exported key, invokes the CLI from the repository root and restores the kernel
  directory. Four synthetic regression cases cover both starting directories and
  both key sources; the non-generating CLI boundary prevents paid calls in tests.
- Token-count API failures persist private diagnostic records. Before generation
  they produce `api_error` with zero generation cost; during summary measurement
  they preserve already-recorded summary costs and produce null-score escalation.
  Head/tail measurement failures also become explicit failed representations.
  Every body count is established in preflight before any paid generation.
- Full installed-package suites: **91 passed in 7.03s on Python 3.11.4** and
  **91 passed in 7.87s on Python 3.12.12**. Both runs include actual offline
  notebook executions. The Python 3.11 environment is separate under `/private/tmp`.
- Ruff passes for the entire project. Public content scan passes across **65**
  candidate files and **175** protected files, with zero findings. Notebook source
  outputs remain empty. No frozen protocol, real generation, commit or push added.
- API key, model IDs, local verified prices, stage financial authorization and
  human review/data-use confirmation are still absent. M3 empirical through M6
  remain pending; no permission is inferred from automatic goal continuation.

### Live-prerequisite blocker audit, 2026-09-06

The previous goal turn made implementation progress (notebook entry and count-failure
fixes). This third consecutive goal turn rechecked the same unresolved live-access
and authorization condition observed in the original implementation turn and the
following continuation. No new user authorization or configuration was supplied.

Current evidence: API key absent from shell/root `.env`; monitor and summarizer IDs
unset in both configurations; local price snapshot absent; data-use and rubric-review
flags false. The 70 real normalized inputs still load, but real run manifests and
call records remain absent, results remain `experiment_not_executed`, and no frozen
protocol exists. There is no live experiment/job to wait on.

The local implementation work and targeted fixes are verified in the preceding
records. Required next work is M3 real inference, followed by reviewed development,
freeze, main execution and empirical analysis. Those steps cannot proceed without
external configuration and the explicit financial/data-use authorization required
by sections 9 and 11 of the supplied plan. The goal is marked blocked awaiting those
inputs, not complete. No additional tests, model calls, commits or publication were
performed merely to extend this continuation.
