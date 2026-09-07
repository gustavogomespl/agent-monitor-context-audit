# Protocol and implementation decisions

## 2026-09-06: hardware eligibility instead of an H100 name check

During guided setup, the author received an RTX PRO 6000 Blackwell Server Edition
with 97,887 MiB, not the originally planned H100. A user-run PyTorch BF16 matrix
check succeeded with compute capability 12.0 and CUDA 13.0. Codex replaced the
literal H100 name check with one-GPU, >=75,000 MiB and native BF16 capability
>=8.0 requirements, retaining actual hardware provenance in every session.
NVIDIA and vLLM documentation support hardware eligibility; Qwen startup and
generation remain untested. This is a pre-pilot infrastructure adjustment, not
evidence about monitoring quality. Differences in GPU hardware must be reported.

## 2026-09-06: acquisition permission-bit compatibility

The author reported an acquisition failure in Colab. A read-only diagnostic
returned Git diff exit code 1 with `core.fileMode=true` and 0 with it disabled,
isolating worktree executable-bit differences. Source verification now ignores
that filesystem metadata for its Git comparison. All pinned-source and content
checks remain in place. Codex reproduced the failure with independent synthetic
Git fixtures; no benchmark scores or model generations informed this fix.

## 2026-09-06: branch-based Colab setup

The author requested selecting the source branch directly in the notebook.
`BRANCH="pilot"` now selects this repository by default; the first setup resolves
and saves an exact commit. Reconnects retain that commit and reviewed frozen
snapshots take precedence, so a later branch update cannot silently change an
experiment. An explicit SHA and the source ZIP remain optional alternatives.
Codex implemented this setup change and checked it using independent synthetic
Git repositories; no experimental scores informed the change.

## 2026-09-06: author-requested Qwen/Colab infrastructure revision

The author explicitly requested adapting the experiment to Qwen and creating a
Colab notebook after the commit review. The implementation selects one pinned
`Qwen/Qwen3.8-27B` for both independent roles, BF16 on one H100, with a short-lived
loopback vLLM server. It retains the original scientific comparison. The legacy
Anthropic path remains optional; no run mixes backends.

Exact body/request counts replace provider differential estimates for Qwen.
Model/tokenizer revision, inference package versions, sampling and thinking
settings are frozen and cached. The pilot first counts all eligible full inputs
without scoring test examples. GPU request time and managed session overhead have
separate numeric fields and a documented allocation convention. Session deadline
and process cleanup are supervised; lost receipts remain reserved until explicit
reconciliation. This is an estimated GPU cost, not a Colab invoice or a guarantee
about allocation outside the supervised block.

The same change fixes four independently reproduced defects: acquisition outside
ignored private roots, ignored frozen bootstrap settings, success exit codes for
incomplete runs, and shrinking CSVs on interrupted resume. These were identified
with synthetic fixtures and code review, not empirical test scores.

Codex implemented and reviewed this revision using parallel AI assistance and
official Qwen/vLLM documentation. The author has authorized the infrastructure
change, not supplied pilot results, reviewed model outputs, finalized data-use
terms, or approved a frozen scientific conclusion. No model generation, GPU
allocation, public push, or local source commit occurred during implementation.

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

No API generation has been executed for this project at this stage. Explicit model
IDs, an authorized execution budget, account/data-use confirmation and live opt-in
are still needed for the empirical milestones. The Qwen amendment below uses GPU
hours; the Anthropic path retains its explicit monetary cap. Keys must remain local
and must never be copied into decision records. A public push/release requires
separate explicit authorization.

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

## 2026-09-06: author-authorized Qwen time budget

During guided Colab setup, the author explicitly stated that USD has no limit but
GPU use has a 12-hour ceiling. For this Qwen path, a cumulative GPU-time allowance
replaces the previously required positive dollar cap/rate. Pilot, development,
test and resumes under the same private workspace share 43,200 seconds; separate
phase directories do not reset the allowance. The notebook starts with one-hour
managed sessions and defaults to disconnecting as soon as managed accounting is
saved. Exports and reports run later on CPU so postprocessing does not extend the
GPU session.

Before the first managed run, the author declares prior GPU setup/idle allocation.
That debit and the original ceiling persist immutably. Additional observed setup/idle
periods can be debited explicitly with stable unique usage IDs; repeating the same
ID and duration does not double-charge the allowance. Each managed session reserves
its maximum seconds before launch, settles known elapsed time after teardown and
retains uncertain usage until explicit evidence-based reconciliation. The watchdog
controls managed process groups only. Platform allocation outside the supervised
block, including later setup/idle periods, needs these explicit usage entries; this ledger
does not certify Colab's complete billable allocation or enforce its billing cutoff.

An effective hourly dollar rate is optional. Without one, exported and reported USD
costs remain null/unavailable; no rate or free-inference claim is invented. Request
latency and managed GPU-time receipts still record resource use. If supplied, the
rate estimates money from duration and is not a verified provider invoice.

Codex implemented this execution-budget amendment at the author's request. It does
not imply review of family grouping, rubric or pilot outputs. The existing private
data boundaries, explicit live opt-in, development-before-test freeze and null-score
failure policy remain in effect. No real model inference or empirical scores were
produced by this amendment.

## 2026-09-06: TorchAudio CUDA build and setup import validation

The author's real Colab startup log identified a binary mismatch: Torch reported
CUDA 13.0 while TorchAudio reported CUDA 12.8. The model server exited during an
import; the log does not demonstrate model loading, memory fit or inference.

Codex checked the tagged vLLM 0.28.0 CUDA requirements (Torch 2.13.0 and TorchAudio
2.11.0), TorchAudio's documented stable ABI compatibility and the official cu130
wheel index, including the Python 3.13 Linux wheel. The notebook now replaces the
mismatched auxiliary wheel with `torchaudio==2.11.0+cu130` only for this verified
Torch/vLLM/CUDA combination. `--no-deps` preserves core runtime pins. A fresh
subprocess imports TorchAudio and the vLLM API server module before setup is marked
ready; it starts no server or model. Before/after package versions are retained in
private setup history. A failed repeated setup clears the previous ready flag.

Synthetic boundary tests cover repair, unchanged core packages, compatible wheels,
failed installs/imports, unsupported combinations and failed repeated setup. They
are not a live verification of the repaired Colab runtime. No protocol settings or
test-set scores informed this dependency correction.

## 2026-09-06: author-requested guided Colab workflow

The author requested a notebook that executes end to end with a friendlier interface.
Codex replaced the separate setup/acquisition/live/analysis flags with a native Colab
form and one guided run. Explicit start, data-use and rubric confirmations remain.
Pilot includes its report; development finishes/reuses the pilot before evaluating
all development pairs. Test is a later explicit selection requiring reviewed
successful development; the existing local freeze validation runs before scoring.
The scientific runner, four conditions, prompt texts and frozen-method checks are
unchanged. Combining ordered setup/freeze/run/report actions in one selected stage
does not authorize automatic progression to the held-out stage.

The notebook embeds checksummed public runtime source and bootstrap scripts to
repair recognized older checkouts without a separate upload or branch push. It
prevalidates paths and hashes, backs up changed source privately, rejects unknown
local edits and protects recorded scientific code hashes and frozen snapshots.
Weights, dependencies and private data are not embedded. The generator and readable
bootstrap scripts are the source of truth; a regression test detects a stale payload.

A notebook allocation receipt measures setup, acquisition and bounded reporting
in addition to the existing supervised model interval. Checkpoints subtract already
charged supervisor time before adding overhead. Existing initial debits are read
from the immutable ledger instead of repeated form edits. A release timer bounds
the active workflow; cleanup also requests release on setup and model failures.
Known failed-setup overhead is recoverable once. Unknown end times still require
verified reconciliation, and Colab allocation outside the workflow remains outside
these measurements. Unknown USD costs remain null. This replaces the earlier
instruction to always perform reporting in a separate CPU-only session; reports
now have a 120-second subprocess deadline within the remaining shared allowance.

Offline synthetic tests cover ordering, consent/review gates, successful reuse,
partial-run stops, cleanup, budget recovery without double charging, and snapshot
integrity. They do not demonstrate a loaded Qwen model or completed scientific study.

## 2026-09-06: Colab entry point and run diagnostics

The author asked to revise the project so the guided Qwen notebook runs intuitively
on Colab. Claude Code changed only the operating surface. The README and the guide
link the notebook committed on the `pilot` branch through Colab's GitHub loader, with
upload as the alternative; the badge serves whatever is pushed. The notebook now checks
the GPU before mounting Drive, so a CPU runtime stops without side effects and names
the runtime-type fix; a stopped run prints the error class and first message line,
recognized hints and the private diagnostic path instead of a generic message; an
incomplete phase prints the runner exit code, successful/expected evaluations, counts
per evaluation status and the worker's final handled error line; the inert default
prints which form fields are unchecked; dependency installation is quiet except for
errors. Pydantic validation details and log excerpts remain only in the private Drive
diagnostic, so notebook output never repeats benchmark text.

The scientific runner, prompts, budgets, consents, freeze gates and pinned versions are
unchanged. The model repository, its exact revision, vLLM 0.28.0 and the public branch
were checked to exist at revision time. Synthetic tests cover the step ordering, the
checklist, error and partial-run diagnostics and the hint boundaries; they are not a
live Colab verification. No model was run and no empirical score was produced.

## 2026-09-06: native sampler for the Qwen managed server

The author supplied a private vLLM 0.28.0 startup log from the SM120 runtime.
Checkpoint loading completed at 50.22 GiB; dummy sampler profiling then failed in
FlashInfer's JIT architecture check. The final engine initialization exception was
a consequence. The log establishes neither an out-of-memory failure nor why
FlashInfer's compilation targets were ineligible. This attempt stopped before the
scientific worker could run, so it provides no evaluation scores.

Codex checked the pinned vLLM implementation and set
`VLLM_USE_FLASHINFER_SAMPLER=0` in the managed server's environment before import.
The native PyTorch sampler is selected consistently across all phases, rather than
switching after an inference failure. Only explicit engine overrides, never the
full inherited environment, are recorded in private session metadata. The source
hash and existing recorded/frozen-method guards also cover this change. Model and
dependency pins, attention selection, BF16, context length, seeds and sampling
parameters are unchanged; bitwise equivalence is not asserted.

The rebuilt guided notebook embeds the correction and recognizes this specific
sampler error with a content-free hint. Existing startup-only workspaces retain
dataset IDs, pins and cumulative time accounting. Synthetic tests cover inherited
environment overrides, provenance without secrets, the hint and notebook integrity.
They do not validate a successful GPU startup, generation or empirical study. No
held-out scores informed the correction.

Sources: [pinned vLLM sampler](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/sample/ops/topk_topp_sampler.py),
[pinned vLLM environment settings](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/envs.py).

## 2026-09-06: recover a token-only context failure before generation

The author supplied a new pilot log showing successful server startup followed by
the worker's context guard: four transcripts exceeded 65,536 tokens. The preflight
stores counts for all eligible inputs before any scoring, as required by the
existing protocol. This is evidence of model startup and tokenization, not of a
completed generation or an empirical monitoring result.

Codex added guided recovery using that complete, matching preflight. Context is
chosen from the largest full monitor/summarizer prompt plus its configured output
cap, adding up to 1,024 tokens of headroom and rounding in 32,768-token increments.
The ceiling remains the pinned model's native 262,144 tokens; there is no RoPE
extension, truncation, pair exclusion, prompt change or sampling change. All phases
receive the same choice. It is permitted only before generation evidence in any
phase and before freezing; test-stage runs cannot trigger it.

The notebook records the count-derived decision, checksum and previous run identity,
then uses distinct context-suffixed phase configs and directories. Existing run
manifests, cached counts, logs and budget receipts are not overwritten. Source and
dataset guards remain intact; this correction changes only the bootstrap and its
generated notebook, so the recorded scientific code hash is preserved. A saved
failure is recovered before startup; a new token-only failure permits one bounded
restart after cleanup and a remaining-time check. No inference error or evaluation
score authorizes automatic retuning.

Synthetic tests cover output allowance, persistent selection, preservation of old
records, incomplete/stale counts, native overflow, generation/freeze gates, bounded
restart, and routing of reports, diagnostics and freeze validation to the active
attempt. They do not establish that the larger window fits the author's GPU or that
generation completes. The author requested focusing on correct execution; this
change does not reset or rewrite time accounting.

Source: [exact model configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json).

## 2026-09-06: author-authorized summary-v2 development amendment

The author supplied private pilot artifacts for offline diagnosis and authorized
improving concision and regeneration while retaining the summary cap and citation
validators. Saved development failures motivated this revision: summaries could
exceed their cap, and a later shorter structured response could omit required
claim-level or identifier-level citations. No held-out scores or transcript text
are used in public prompts, documentation or tests.

Codex changed the generic prompts for both summary conditions to target 60–80% of
the same per-example budget, whose maximum remains 1,024 tokens. The permitted
regeneration restates length, format, schema and citation requirements together
and receives the original model-visible input. It receives neither the rejected
candidate nor validation error text, labels or monitor feedback. The maximum is
still two attempts total; all attempts are accounted for. Existing validators,
null-score failures and the four conditions remain unchanged. This is a prompt
and regeneration development amendment, not evidence of improved performance.

The notebook's hidden `EXPERIMENT_VERSION = "summary-v2"` selects the local checkout
`/content/agent-monitor-context-audit-summary-v2`. Its source/configuration provenance
lives under Drive `versions/summary-v2/`, including source pin, context selection,
Git metadata, source receipts and any later freeze. First setup copies the parent
source pin, measured context and public manifests; reconnects do not overwrite
the version's own copies. First creation is blocked by an already frozen parent
or existing test run. The legacy source guard continues to protect all runs;
only the explicit new version scopes its source guard to its own run manifests.

With the inherited window, the new pilot uses
`runs-private/qwen-pilot-summary-v2-ctx196608`, exports to
`numeric-results/summary-v2/pilot`, and writes workflow diagnostics under
`runs-private/notebook-status/summary-v2`. Pilot and development configurations
record `development-summary-v2`; the reviewed held-out freeze keeps the existing
`protocol-v1` contract. All 24 pilot evaluation units start afresh, including the
previously successful conditions, so old calls and results are not mixed into
the revised methods. Earlier caches, configs, reports and source remain intact.

The parent model/tokenizer/runtime pin, private dataset, opaque IDs, family split,
context choice and cumulative GPU ledger are shared. There are no new Drive
symlinks, copied data or second GPU allowance. The user keeps the same Drive folder
and existing start/data/rubric confirmations. The canonical notebook is rebuilt
from source and delivered as `dist/qwen_colab_summary_v2_<hash>.ipynb` for upload;
no public push or live generation is performed by this implementation.
The embedded source explicitly includes the amended plan, this decision record and
the operating guide so the eventual Colab freeze retains the revised methods.

Synthetic tests exercise the prompt/repair contract, strict validation, isolated
source/config/run/report routing, prior-artifact preservation and shared budget.
These checks do not establish completion of the revised GPU pilot, better coverage
or an empirical monitoring result. Full development still precedes reviewed test.

## 2026-09-07: author-authorized summary-v3 schema and citation amendment

The author authorized another development version after offline diagnosis of two
remaining citation-validation failures in the summary-v2 pilot. These were saved
development failures; held-out scores did not motivate the change. No benchmark
text, private example identifiers, annotations or canaries are included here or
in public prompts and tests.

For Qwen, `structured_summary_mode="schema_citations_v1"` selects constrained
JSON-schema decoding for the structured summarizer. The internal generated draft
has four claim fields, each an array of objects with required string `text` and
nonempty `evidence_event_ids`. Reference values are restricted to an enumeration
of that transcript's visible IDs; additional properties are forbidden. There is
no global draft index and no cited-string regular-expression constraint.

The renderer rejects empty or blank text locally, preserves accepted item text
and appends bracketed citations from only
the references the model selected for that item. It produces the existing four
arrays of cited strings and derives the fifth field, `source_event_ids`, as the
sorted union of selected references. Rendering neither adds nor removes claims,
infers evidence, invents references or repairs meaning. Raw drafts are retained
privately, so the model's selections and final representation remain inspectable.
The unchanged final validator rejects unknown IDs and uncited claims, verifies
the complete five-field representation and measures its full serialized JSON,
including the assembled ID list, under the existing per-example cap of at most
1,024 tokens. No summary is silently truncated to pass validation.

Actual CPU checks of XGrammar 0.2.3 showed that a cited-string pattern was not a
reliable per-item constraint: a neighboring cited item could permit an uncited
item. The final design uses explicit typed references on each claim instead.
Only summary-v3 requires the explicit `xgrammar==0.2.3` pin and performs a grammar
compiler/matcher check before model weights are loaded. Independent synthetic
valid drafts are accepted, while missing, empty and unknown references are rejected
even beside cited neighbors; missing text and extra properties are also rejected.
This check exercises the actual decoder backend on CPU without model generation.
The mechanism uses [vLLM 0.28.0 structured outputs](https://docs.vllm.ai/en/v0.28.0/features/structured_outputs/).

The raw generation allowance remains 1,600 tokens and there are at most two
attempts, with both recorded and charged. Both summary conditions retain the
60–80% target and common evidence priorities introduced in summary-v2. The allowed
regeneration receives the original input and the required constraints; it does not
receive the rejected response, validation error text, labels or monitor decisions.
The monitor, failure/null-score policy, four conditions and family split retain
their contracts. Earlier prompt-only structured-summary mode remains readable.

This amendment changes the whole structured-summary intervention, including the
decoder and deterministic assembly. A future difference between conditions cannot
be attributed solely to JSON syntax. Requiring explicit citation references does not prove
that a statement is faithful to the cited evidence. The final validator and later
human assessment retain their distinct roles.

The notebook hides `EXPERIMENT_VERSION="summary-v3"` in its implementation and uses
`/content/agent-monitor-context-audit-summary-v3` for local source. Drive
`versions/summary-v3/` isolates source/configuration provenance and Git metadata.
On first setup it inherits the available summary-v2 source pin/context, falling
back to the legacy selection, and retains its own copies on reconnect. Creating
the version is blocked by a frozen or test-run workspace in earlier versions or
the legacy path. Old source, configurations, caches, reports and numeric results
remain intact; this does not authorize rewriting any frozen protocol.

With the inherited context, the new pilot uses
`runs-private/qwen-pilot-summary-v3-ctx196608`, saves outputs under
`numeric-results/summary-v3/pilot`, and writes diagnostics to
`runs-private/notebook-status/summary-v3`. All 24 pilot units start afresh, including
previously successful conditions. Later resumptions can use this version's own
incremental records, without mixing earlier-version responses into the new pilot.
The original Drive root still supplies the shared private dataset, opaque IDs,
family split, model/tokenizer/runtime pins and cumulative GPU ledger. There is no
new 12-hour allowance.

Codex implemented this author-requested amendment and the notebook is delivered
as `dist/qwen_colab_summary_v3_<hash>.ipynb`. Source and mode provenance distinguish
it from summary-v2. Offline validation is not GPU validation of constrained decoding
or evidence of a successful revised pilot or improved monitoring performance.
Explicit start/data-use/rubric confirmations and full development before reviewed
test remain required. No publication, public push or live generation is performed
by this implementation.

## 2026-09-07: author-authorized summary-v4 compressed-budget amendment

The author supplied private summary-v3 development pilot records with 23 of 24
evaluation units successful and authorized increasing the compressed maximum to
2,048 tokens. The remaining structured-summary failure exceeded the 1,024-token
cap in both assembled attempts: 1,542 tokens in one and 1,124 tokens in the final
attempt. These development length-validation diagnostics motivated the change;
held-out scores did not. No benchmark text, private example identifiers,
annotations or canaries are included in public documentation, prompts or tests.

Summary-v4 uses the common per-example ceiling
`B_i = min(T_i, max(128, min(2048, floor(0.25 * T_i))))` for `head_tail`,
`free_summary` and `structured_summary`. Only the maximum rises from 1,024 to
2,048; the token fraction remains 0.25 and the minimum remains 128. The common
initial task and monitor policy stay outside the body cap, while all fields,
citations and omission markers count toward it. Short inputs retain the existing
identity behavior. `full` remains integral and is never silently truncated.

Both summary conditions receive a 3,200-token raw generation allowance, preserving
the previous 1,600-to-1,024 ratio to the maximum representation cap. The raw
allowance is distinct from the final validated representation ceiling. The
60–80% target, common evidence priorities, two-attempt maximum and accounting of
both attempts remain unchanged. Monitor output retains its 700-token allowance.
Persistent failures keep null scores and escalate to review.

The v3 `schema_citations_v1` typed draft, constrained decoding, `xgrammar==0.2.3`
pin and CPU grammar check, deterministic assembly and final validators remain
unchanged. The assembled structured JSON, including `source_event_ids`, must fit
the new per-example cap. Rendering still neither changes claims nor invents
evidence. Regeneration receives the original input and generic requirements,
without rejected responses, validator error text, labels or monitor decisions.
Typed citations do not establish that a claim is faithful to its cited evidence.

The hidden `EXPERIMENT_VERSION="summary-v4"` selects the local source checkout
`/content/agent-monitor-context-audit-summary-v4` and Drive `versions/summary-v4/`
for isolated source/configuration provenance, Git metadata and any later freeze.
First creation inherits the newest valid parent source pin and measured context
from summary-v3, then summary-v2, then legacy; reconnects retain v4's own records.
Any earlier frozen workspace or test-run evidence blocks version creation.
Earlier source, configurations, caches, calls, reports and numeric results remain
intact and are never reused as new-version responses or mixed into v4 scores.

All 24 pilot units start afresh. With the inherited 196,608-token context, the
pilot uses `runs-private/qwen-pilot-summary-v4-ctx196608`; outputs go to
`numeric-results/summary-v4/pilot` and diagnostics to
`runs-private/notebook-status/summary-v4`. Development configurations identify
`development-summary-v4`. The original Drive root supplies the same private
dataset, opaque IDs, family split, model/tokenizer/runtime pins and cumulative
GPU ledger. The measured context is inherited, not retuned from generation
failures, and request-token preflight still validates the larger output allowance.
The 12-hour allowance is shared with all prior versions and is not reset.

This changes the compression intervention across all compressed conditions.
Larger outputs permit less compression and may increase generation time, monitor
latency and GPU use; equal caps do not imply equal realized lengths. Earlier pilot
results document development history and cannot establish v4 performance. Full
development, author review and the existing explicit test freeze remain required.

Codex implemented this author-requested amendment with parallel AI assistance.
The notebook is delivered as `dist/qwen_colab_summary_v4_<hash>.ipynb` and embeds
the amended plan, decision record and operating guide. Offline checks are not a
guarantee of successful GPU execution, completion within the remaining allowance,
improved coverage or monitoring quality. Existing start/data-use/rubric
confirmations remain required. No GPU inference, paid API call, commit, public
push or publication is performed by this implementation.

## 2026-09-07: author-authorized summary-v5 bounded drafts and short-input budget

Private v4 development completed 61 of 64 units. Two structured outputs exceeded
their final budgets; another invalid response used its full raw output allowance
without a retained finish reason. The author authorized three operational changes:
control structured draft length, revise the short-input allowance and preserve
safe termination diagnostics. These development failures motivate v5; held-out
scores do not. No historical missing reason is reconstructed as truncation.

The common body ceiling becomes
`B_i = min(T_i, max(1024, min(2048, floor(0.25 * T_i))))`. All three compressed
conditions receive that ceiling. Bodies at or below 1,024 tokens use the full
body without summary calls, extending the existing identity rule. The raw summary
allowance remains 3,200 for both summaries, monitor output remains 700, and both
summary attempts retain the original input, common evidence priorities and
60–80% target. No rejected output, validation error, label or monitor decision
is supplied as retry feedback.

The structured mode is `schema_citations_bounded_v1`: at most two claim objects
per field, at most two selected references per object, and `floor(B_i / 4)`
JSON text units per claim. This reserves space structurally across the four
fields instead of relying on the length instruction alone. A literal Unicode
character or JSON escape is one unit, with escaped surrogate pairs using two.
The explicit escape-safe pattern is specific to XGrammar 0.2.3's serialized
string matching. Its `minLength`/`maxLength` implementation does not preserve
valid escaped strings, so those keywords are not used. The production schema is
compiled and exercised at the minimum/maximum budgets on CPU before model startup.
See the [pinned converter source](https://github.com/mlc-ai/xgrammar/blob/v0.2.3/cpp/json_schema_converter.cc)
and [vLLM structured output interface](https://docs.vllm.ai/en/v0.28.0/features/structured_outputs/).

Structural bounds do not guarantee an exact token count. Local validation checks
original serialized text units and rejects excess claims/references or decoded
text length; the unchanged final validator
counts all assembled fields/citations against the exact token ceiling. Assembly
preserves claim text and only indexes model-selected evidence. No claim is dropped,
sliced or repaired. This bounded intervention may omit evidence and must be
evaluated as a complete method, not as a causal JSON-format effect. Free-summary
decoding and the monitor rubric are unchanged. Persistent failures remain null
and escalate after at most two attempts.

New private call records preserve allowlisted `finish_reason`, the existing
`stop_reason` alias, bounded `provider_stop_reason` metadata, and categorical
`diagnostic_cause`. Raw stop strings are redacted and no reasoning/tool response
content is retained. Usage equal to the raw allowance alone does not imply a
reported length termination. Existing cached records remain unchanged.

All 24 pilot units are fresh in isolated summary-v5 calls/configs/results. First
setup inherits v4, then v3, v2 or legacy source/context provenance. Existing model,
runtime, data, opaque IDs, family split, context window, source pins and remaining
shared 12-hour ledger survive. Earlier caches are not reused as new responses;
earlier results are not overwritten. Freeze/test evidence in any prior version
blocks version creation. Full development and explicit reviewed test freezing
remain necessary.

Codex implemented this amendment with parallel AI assistance. Validation uses
independent synthetic fixtures, actual CPU grammar checks and private offline
migration of saved control artifacts. No benchmark content is added to tests or
public files. These checks are not real v5 inference, evidence of improved scores,
or a guarantee of GPU completion. Delivery is
`dist/qwen_colab_summary_v5_<hash>.ipynb`, with empty execution outputs. No live
generation, commit, public push or publication is performed by this change.
