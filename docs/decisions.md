# Protocol and implementation decisions

**2026-09-05 — initial implementation; not frozen.** The user supplied
[research_plan.md](../research_plan.md) and authorized building the project from
that plan. It remains the scientific scope. The existing Apache-2.0 license is
preserved. No publication or paid experiment is implied by implementation
authorization.

| Decision | Reason / status |
| --- | --- |
| Retrospective external monitor; no new acting agent. | Matches the proposed estimand. No transcript commands or model execution tools. |
| One official Anthropic API client. | Minimal provider scope; explicit model IDs remain user-configured and pilot-validated. |
| All conditions omit thinking. | Uses only visible text and tool interactions and does not reproduce upstream thinking/stripping variants. |
| Fixed initial task outside compression. | Preserves the observable request without consulting evaluator scenario annotations. |
| Typed label-free input and separate evaluator labels. | Reduces direct label/metadata leakage. Event contents can still contain naturally informative evidence. |
| Four conditions, common priorities, per-example ceiling. | Implements the supplied design; equal realized lengths and pure JSON effects are not assumed. |
| Whole-family split, including both pair members. | Prevents obvious related-case overlap. Grouping uncertainty must be recorded before test. |
| Score threshold 50; technical failure escalates with null score. | Exploratory operational convention, not calibration; no benign defaults on failure. |
| Primary common complete-pair set; paired family bootstrap. | Keeps comparisons on the same cases; report selection/coverage limitations. |
| Budget and retry ledger persists incrementally. | Resume must retain all billable attempts and preserve intentional repetition identity. |
| No real scores until an authorized live run. | Synthetic fixtures validate engineering only; they cannot fulfill empirical milestones. |

The initial proposed compression values are 25%, minimum 128 and maximum 1024;
the threshold is 50; proposed bootstrap repetitions are 2,000 and the coverage
quality threshold is 95%. They are development-stage defaults, not validated
constants. The actual frozen config and hashes, rather than this prose, identify
the eventual protocol version.

## Inventory decisions made before scoring

The exact upstream pin is `218c58315cc01ff0dc5a100e906c27d82d259521`.
The observed source contains more directories and categories than the README's
aggregate description. The strict v0 inventory retains 35 pairs, not the
anticipated 37: it excludes three whole multissession cases spanning six
directories, one unpaired case and two malformed pairs whose benign session
lacks required cwd information. No source records were repaired or invented.
The [sanitized manifest](../data/manifests/inventory.json) records opaque reasons.

Seed `20260905` assigns eight pairs to development and 27 to test while keeping
33 inferred families intact. Family relations use stage/session structure,
identical requests/scenario strings and explicit sidecar cross-references,
transitively. This is an implemented conservative grouping proposal; the author's
family review is outstanding. Body-size inventory is in UTF-8 bytes only, and
model-specific token/context feasibility is still unverified. These are factual
inventory results, not monitor-performance evidence or a completed M3 pilot.

Acquisition verifies decoded source artifacts against the pinned encrypted
payloads. The inventory records normalized content hashes and evaluator-label
integrity; load-time checks reject changed private derived artifacts. An aggregate
normalized-content digest binds the public provenance to the private inputs
without exposing their contents.

The official Anthropic SDK resolves to version 1.4.0 in the initial lockfile.
Application-controlled attempts disable hidden SDK retries. Pre-request financial
reservations use a conservative context-window upper bound, which can stop a
request earlier than an optimistic estimate. Unknown charges are labeled bounds.
Caps apply cumulatively within a run directory; separate stages need explicit
budget allocation. Planned result-manifest units, not just observed rows, are
required to verify primary coverage.

## Freeze procedure and amendments

Complete the real development pilot, inspect cost/failures/lengths, and obtain
author review of the hypothesis, rubric and pilot examples. Record chosen models,
token method/context limits, grouping/split, prompts, generation settings, budget
rule, threshold, metrics, failure policy and price provenance. Freeze their hashes
with the code revision in a `protocol-v1` commit/tag before any test scoring.
Creating a freeze manifest does not assert that a Git commit/tag or pilot already
exists. Record those separately. Do not tune prompts against test scores. Any
test-informed change starts a documented exploratory version and must not be
presented as an untouched holdout evaluation.

## AI assistance and human ownership

Codex assisted with repository implementation, tests, technical documentation,
source verification and proposed operational details. This assistance does not
constitute independent scientific replication or human approval of decisions.
The author has not yet reviewed the implemented monitoring rubric, any real pilot
outputs or empirical conclusions. These reviews remain required. No performance
finding is attributed to the author in advance. No LASR application answers were
written or reviewed as part of this project.

No API generation has been executed for this project at this stage. Model IDs,
an explicit monetary cap, account/data-use confirmation and live opt-in are still
needed for the empirical milestones. Keys must remain local and must never be
copied into decision records. A public push/release requires separate explicit
authorization.

## Local verification environment

The macOS sandbox prevented uv from reading system network configuration; dependency
resolution ran with approved network access. Python 3.12.12 and the exact package
versions are recorded in `uv.lock`. This machine also marked the editable install's
`.pth` file with macOS `UF_HIDDEN`, which Python ignores. Source checks therefore
use `PYTHONPATH=src`; a final non-editable installation checks the distributed package.
This is local environment behavior, not an empirical model result.

Private model-call latency measures summed generation requests, including retries;
it excludes token-count preflight and local processing. Cached reanalysis records
zero new generation cost and retains the original run's measured generation latency.

## 2026-09-06: notebook entry and count-failure behavior

The optional live notebook resolves the project root independently of the kernel's
working directory. Root `.env` loading happens only inside explicit `RUN_LIVE=true`
execution and never overwrites exported credentials. The original kernel directory
is restored after the CLI finishes.

A provider token-count error is not a zero-token observation. Full-body counts are
preflight requirements; a failed preflight stops before paid generation. A later
request/output measurement failure is recorded as `api_error`; an already-produced
summary keeps its measured generation cost even when its representation cannot be
accepted. No extra summary is generated merely to repair a failed counter request.
These changes were prompted by independent synthetic reproductions, not test-set
monitor scores. Real pilot and author review remain outstanding.
