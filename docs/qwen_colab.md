# Qwen3.8-27B on Colab

## Run in four steps

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gustavogomespl/agent-monitor-context-audit/blob/pilot/notebooks/03_qwen_colab.ipynb)

1. Upload the supplied `dist/qwen_colab_summary_v7_<hash>.ipynb` with
   **File → Upload notebook**, or upload the rebuilt `notebooks/03_qwen_colab.ipynb`.
   The badge above loads the committed notebook from `pilot`; it reflects this
   version only after the rebuilt notebook is published.
2. Select one eligible GPU runtime with **Runtime → Change runtime type**:
   H100 80GB or RTX PRO 6000 Blackwell 96GB.
3. Leave **STAGE = pilot** for the first run. Confirm data use and rubric review,
   then check **START_RUN**. Keep the optional workspace name unchanged to resume
   the existing private data and budget. The notebook selects the new development
   version automatically. The test review checkbox is only needed for test.
4. Select **Runtime → Run all** and allow Google Drive access when asked.

The notebook handles the GPU check, source preparation, dependency installation and
import checks, acquisition/inventory, supervised generation, numeric export and
reporting. It saves the report to Drive and releases GPU allocation at the end,
including setup/error paths. Default Run All without confirmed form fields prints
the unchecked fields and does not mount Drive or execute an experiment.

## Current development amendment: summary-v7

Upload the supplied v7 notebook, keep **STAGE = pilot** and the same Drive folder,
confirm data use and rubric review, then check **START_RUN** and select **Run all**.
The header must show `Experiment: summary-v7`. All 24 evaluations are fresh v7
calls; neither successful nor failed v6 evaluations become v7 results. The test
review checkbox remains false until a separate, reviewed test execution.

The supplied v6 pilot completed 24 of 24 evaluations. Full development completed
63 of 64; both attempts of its remaining structured-summary failure put an event
ID in claim prose without selecting it in that item's evidence list. The local
validator rejected those drafts before the monitor. V7 addresses this format
failure and preserves the v6 results, including the null failure score. It does
not establish improved detection or a successful new GPU run.

`schema_citations_separate_ids_v1` puts IDs only in each item's
`evidence_event_ids`, with no event IDs in claim prose. The model must select all
visible evidence needed to support the claim. Decoder constraints and the local
validator enforce this separation. At most two claims per field and two selected
references per claim remain allowed. Nonblank text, valid visible references and
all typed fields remain mandatory. The application preserves accepted prose and
adds only the citations chosen by the model. It never fills in a missing reference,
rewrites a claim or removes text to make an invalid output pass. Valid citation
format is separate from whether an event supports the claim.

Claim prose uses literal non-control Unicode characters. Normal JSON escapes for
quotes, backslashes and controls remain supported. This canonical serialization
prevents Unicode escapes from hiding an event ID; it adds no text-length ceiling
and does not rewrite accepted decoded prose. Ordinary identifiers containing a
similar substring remain valid when they are not standalone event-ID tokens.

The common final body rule remains
`B_i = min(T_i, max(1024, min(2048, floor(0.25 * T_i))))`. At or below 1,024 tokens,
all compressed conditions reuse the body unchanged. Both summarizers retain the
60–80% target, 3,200-token raw allowance and two-attempt maximum. No per-claim
text-length bound is restored. Every assembled field and citation must fit the
exact final token ceiling; failures keep null scores. Full history remains
complete. Retry inputs contain only the original input and generic requirements,
without rejected drafts, validator feedback or evaluator labels.

The v6 monitor mode, rubric, continuous 0–100 scores, 700-token monitor allowance,
visible-evidence ID constraints and all model/runtime pins stay unchanged. Before
weights load, the semantic decoder probe and tokenizer-only full-vocabulary gate
check the v7 production schema. The gate explicitly selects the v7 mode and its
receipt must identify that mode and `two_fast_confirmations_v1`. If an initial
mask measurement exceeds 0.25 seconds, two fresh matchers and bitmasks rebuild
the same accepted-token prefix for independent confirmation measurements. Both
confirmations must meet the same 0.25-second limit; either slow confirmation stops
setup without fallback. All raw timings are saved, and
`DECODER_LATENCY_OK` prints the raw maximum and confirmed gate maximum separately.
The 120-second profile and 150-second worker deadlines remain unchanged. This
handles an isolated CPU scheduling spike while retaining the latency threshold;
v6 keeps its original single-measurement policy. Setup also verifies a finite
confirmed maximum within the unchanged limit. A passed CPU gate does not
validate GPU throughput or monitoring quality. Its private receipt is
`versions/summary-v7/configuration/decoder-latency.json`.

With the inherited 196,608-token context, the new run directory is
`runs-private/qwen-pilot-summary-v7-ctx196608/`, and the report is
`numeric-results/summary-v7/pilot/reproduced/findings.md`. First setup inherits v6
source/context provenance when valid, then v5, v4, v3, v2 and legacy. The dataset,
IDs, split, pins and remaining shared 12-hour budget are retained. Earlier source,
configuration, responses and results stay intact. An earlier freeze or test run
blocks creation of this development amendment. No folder deletion, manual patch
or branch edit is needed.

After a complete v7 pilot, inspect its report and select **development** to run all
eight development pairs. Review that full run before choosing test. This
exploratory structural correction was authorized after development format failures;
held-out outcomes and score optimization did not guide it. See the
[v7 protocol amendment](../research_plan.md) and [decision record](decisions.md).
The badge continues to load the published branch, so upload the supplied v7
artifact until that version is published.

## Historical development amendment: summary-v6

**Execution follow-up (2026-09-08):** supplied v6 artifacts later confirmed a
24-of-24 pilot and 63-of-64 full development. The remaining failure motivated the
v7 amendment above. The following describes the original v6 contract and its
pre-run expectations, not the current notebook.

Use the supplied v6 notebook with **STAGE = pilot** and the same Drive workspace.
Confirm data use and rubric review, then **START_RUN**; all three are initially
false. The test-review checkbox also remains false and is only needed for test.
The header must show `Experiment: summary-v6`. All 24 pilot evaluations are fresh
v6 calls. Review the complete pilot before selecting full development; held-out
test remains a separate reviewed and frozen stage.

The supplied v5 pilot had six structured-summary failures after 12 `ReadTimeout`
attempts at 300 seconds, plus one head/tail monitor citation failure. V6 addresses
these development failures; its GPU success and monitoring quality remain untested.

`schema_citations_compact_v1` retains at most two claims per field and two selected
visible-event references per claim. It removes v5's per-claim text-length pattern
and local text-length bounds. Nonblank text, valid references, deterministic
assembly and the exact final token validator remain required. Removing the text
bound allows longer claims; it does not guarantee that the final summary fits.
No claim is cut or rewritten to fit. Failed final outputs retain null scores and
escalate after at most two attempts.

The shared body rule remains
`B_i = min(T_i, max(1024, min(2048, floor(0.25 * T_i))))`, including the unchanged
identity path for bodies of at most 1,024 tokens. Both summary conditions keep
the 60–80% target and 3,200-token raw allowance. Full history stays complete and
monitor output remains capped at 700 tokens. Retries receive the original input
and generic requirements, without rejected drafts, validation errors or labels.

`monitor_output_mode="schema_visible_evidence_v1"` constrains evidence IDs to those
actually visible in the representation being monitored, for every condition and
both attempts. Empty visible evidence permits an empty evidence list only. The
existing rubric, continuous 0–100 score and other fields remain unchanged. The
model chooses the evidence; the application does not invent or repair citations.
An allowed ID can still support an incorrect conclusion, so decoder validity is
not evidence of faithful monitoring. Older versions keep their original modes.

Before downloading or loading weights, setup verifies the decoder semantically
and checks token-mask latency using the pinned tokenizer/config and its full
vocabulary. This CPU check exercises real token acceptance, mask filling and
Unicode with production summary and monitor schemas. Each mask must take at most
0.25 seconds; any failure stops setup. It does not load weights or generate model
text and does not validate GPU throughput. `DECODER_LATENCY_OK` reports vocabulary
size and the maximum measured mask time. The private receipt at
`versions/summary-v6/configuration/decoder-latency.json` and setup history record
the revision, schema hashes and timings. Reconnects repeat the check without
requiring identical timing values.

Drive paths are `versions/summary-v6/`,
`runs-private/qwen-pilot-summary-v6-ctx196608/`, `numeric-results/summary-v6/pilot/`
and `runs-private/notebook-status/summary-v6/` when inheriting the measured
196,608-token context. The report is
`numeric-results/summary-v6/pilot/reproduced/findings.md`. First setup prefers
v5 source/context provenance, then v4, v3, v2 and legacy. The same data, IDs,
family split, model/tokenizer/runtime pins and remaining 12-hour budget are used.
Earlier source, configs, caches and results stay intact and are not v6 responses.
Any earlier freeze or test evidence blocks creating this development amendment.

See the [v6 protocol amendment](../research_plan.md) and [decision record](decisions.md)
for the scientific scope, failure limits and AI assistance. The Colab badge loads
the published branch and can still point to an older notebook until v6 is published;
upload the supplied v6 artifact to use the local reviewed version.

## Historical development amendment: summary-v5

This section records the v5 design and status when implemented. V6 supersedes its
active paths, per-claim text bounds and monitor decoding mode; v5 records remain
unchanged.

Use the supplied new notebook, keep **STAGE = pilot** and the existing Drive
workspace, confirm the three initial checkboxes, then select **Run all**. The
header must show `Experiment: summary-v5`. Uploading the notebook is sufficient;
manual source patches and branch changes are unnecessary. All 24 pilot units are
new v5 calls. After a complete pilot, review its private report before selecting
development. A complete pilot is not a completed study.

The common body ceiling is
`B_i = min(T_i, max(1024, min(2048, floor(0.25 * T_i))))` for all three compressed
conditions. The higher minimum gives short histories more space; bodies of at most
1,024 tokens remain complete without generating a summary. The maximum remains
2,048, both raw summary allowances remain 3,200, and `full` remains integral.

Structured generation now uses `schema_citations_bounded_v1`: at most two claims
per field, two references per claim and `floor(B_i / 4)` JSON text units per claim.
This means 256 units at a 1,024-token budget and 512 at 2,048. A literal character
or JSON escape is one unit; an escaped surrogate pair is two. The pinned decoder
enforces these bounds while producing the draft. It supports escaped quotes,
backslashes and Unicode, and is checked on CPU before model weights load.
The string pattern depends on XGrammar 0.2.3's serialized-string semantics;
it is not a portable substitute for a generic decoded-string schema validator.

These structural limits reduce available output space; they do not guarantee that
the final representation fits its exact token ceiling or preserves all useful
evidence. The final counter/validator remains authoritative. Assembly never cuts
claims or changes their wording. Both attempts are recorded; persistent failures
remain null and stop advancement. The free-summary decoder, evidence priorities,
monitor rubric and retry-input policy remain unchanged.

New private call records retain `finish_reason`, `provider_stop_reason` and
`diagnostic_cause` even when an output is invalid. The existing `stop_reason` alias
also records the finish category. Raw stop strings are redacted. `length` explicitly
identifies a reported length stop; a completion count equal to the allowance alone
does not. Old cached responses keep their original diagnostics.

Drive paths are `versions/summary-v5/`, `runs-private/qwen-pilot-summary-v5-ctx196608/`,
`numeric-results/summary-v5/pilot/` and `runs-private/notebook-status/summary-v5/`
when inheriting the measured 196,608-token window. The report is
`numeric-results/summary-v5/pilot/reproduced/findings.md`. Existing data, IDs,
family split, model/runtime pins, measured context and the remaining cumulative
12-hour budget are retained; prior results are preserved. First setup prefers
v4 provenance, then v3, v2 and legacy. Any earlier freeze or test evidence blocks
creation. V4 pilot/development scores cannot be reused as v5 results.

The author authorized this exploratory version after v4 development length/output
failures. It has offline synthetic verification; GPU success and monitoring
performance require the new run. See the [v5 protocol amendment](../research_plan.md)
and [decision record](decisions.md) for the scientific scope and AI assistance.

## Historical development amendment: summary-v4

The author authorized summary-v4 on 2026-09-07 after the private summary-v3
development pilot completed 23 of 24 evaluation units successfully. The remaining
structured-summary failure was length validation: its final assembled attempt
measured 1,124 tokens against a 1,024-token cap; the other attempt measured 1,542.
These development diagnostics, not held-out scores, motivated the change. The
author requested increasing the compressed-representation maximum to 2,048 tokens.

All three compressed conditions (`head_tail`, `free_summary` and
`structured_summary`) use the same per-example cap:

`B_i = min(T_i, max(128, min(2048, floor(0.25 * T_i))))`

The token fraction remains 0.25 and the minimum remains 128. The initial task and
monitor policy remain outside this body budget. Every serialized field and citation
in a structured summary counts toward the cap. `full` remains the integral
normalized body; it is never silently truncated. Equal caps do not imply equal
realized lengths, and short inputs retain the existing identity behavior.

Both summary conditions keep the 60–80% target and common evidence priorities.
Their raw generation allowance is 3,200 tokens each, preserving the previous
1,600-to-1,024 ratio between the generation allowance and maximum representation
cap. This allowance is separate from the final representation limit. There are
still at most two attempts, with both recorded and charged; monitor output remains
capped at 700 tokens. Persistent validation failures keep null scores and escalate
to review.

The v3 typed citation draft, `schema_citations_v1` decoding, XGrammar pin and CPU
check, deterministic assembly and final validators are retained. No generated
claims are added, removed or repaired to fit the budget. Regeneration still uses
the original input and generic requirements, without the rejected response,
validator error text, labels or monitor decisions. The historical v3 section below
describes this mechanism and its original, smaller budgets.

The hidden setting `EXPERIMENT_VERSION = "summary-v4"` selects
`/content/agent-monitor-context-audit-summary-v4` and isolated Drive provenance in
`versions/summary-v4/`. First setup inherits the newest valid parent source pin
and measured context in order: summary-v3, summary-v2, then legacy. Reconnects keep
this version's own records. Creating the version is blocked if any earlier version
or legacy workspace is frozen or has test-run evidence.

All 24 pilot units start afresh across the same three development pairs and four
conditions. Earlier successful responses and numeric scores are not reused or
mixed into v4; old source, configurations, caches and results stay intact. Model,
tokenizer and runtime pins, private dataset, opaque IDs, family split, measured
context and the remaining shared 12-hour GPU allowance are retained. Full
development and an explicit reviewed freeze still precede held-out evaluation.

The larger cap permits less compression and may increase generation time, monitor
latency and GPU use. It changes the compression intervention; it does not establish
better coverage, faithful citations or improved monitoring performance. Codex
implemented this author-requested amendment with parallel AI assistance. Offline
checks do not guarantee that the revised GPU pilot will succeed or finish within
the remaining allowance. Existing start, data-use and rubric confirmations apply.

## Historical development amendment: summary-v3

This section records the v3 design and its status when implemented. The v4
amendment above supersedes its representation and raw generation limits and active
version paths; the v3 records remain unchanged.

The author authorized summary-v3 on 2026-09-07 after offline diagnosis of two
remaining citation-validation failures in the summary-v2 development pilot.
Qwen now selects `structured_summary_mode="schema_citations_v1"` for structured
summaries. Constrained JSON-schema decoding produces an internal draft with four
claim fields, each an array of objects containing required string `text` and a nonempty
`evidence_event_ids` array. Each selected ID must come from that transcript's visible
IDs. Both properties are required and extra properties are rejected. The draft has
no global citation index and does not use a cited-string regular expression.

The renderer rejects empty or blank claim text locally, preserves accepted text
and appends bracketed citations using
only the references the model selected for that item. This produces the existing
four arrays of cited strings and adds `source_event_ids` as their sorted union.
It neither adds nor removes claims, infers evidence or repairs meaning. The raw
draft is retained privately alongside the assembled representation.

Summary-v3 pins `xgrammar==0.2.3` for this decoder. Before loading model weights,
setup compiles and exercises the actual grammar on CPU with independent synthetic
inputs. Valid claims pass; missing, empty or unknown references fail, including
when a cited neighboring item is present. This verifies the decoder constraint
without generating a model response. See [vLLM 0.28.0 structured outputs](https://docs.vllm.ai/en/v0.28.0/features/structured_outputs/).

The final validator remains unchanged: it rejects unknown IDs and uncited claims,
checks the full five-field representation and measures all serialized JSON,
including the assembled ID list, against the existing cap of at most 1,024 tokens.
Both summary conditions retain the generic 60–80% target and the same evidence
priorities from summary-v2. The raw generation allowance remains 1,600 tokens and
the limit remains two attempts total. The permitted regeneration uses the original
input and reinforces the requirements without the rejected candidate, validator
error text, labels or monitor feedback. No oversized summary is silently shortened.

This changes the full structured-summary intervention, including its decoder and
deterministic assembly. It does not isolate a causal effect of JSON syntax. Schema
compliance also does not establish that a claim is faithful to its cited evidence.
The legacy prompt-only mode remains readable for earlier summary-v2 records.

The hidden setting `EXPERIMENT_VERSION = "summary-v3"` selects an isolated source
checkout at `/content/agent-monitor-context-audit-summary-v3` and new run identities.
The pilot starts all 24 evaluation units afresh across the same three development
pairs and four conditions; it does not reuse the previous pilot's successful or
failed calls. The model/tokenizer/runtime pins, dataset, opaque IDs, family split,
selected context and cumulative GPU allowance are retained. Existing runs, caches,
configs and reports remain available for comparison. Later reconnects can resume
this version's own incremental records.

This amendment was motivated by development validation failures, not held-out
scores. Creating the version stops if a legacy or earlier version workspace is
already frozen or contains test-run evidence. Existing data-use/rubric confirmations,
explicit start and development review before test still apply. Implementing the
schema mode and offline checks do not establish successful GPU execution, improved
live coverage or better monitoring performance; the authorized pilot must test this.

## What the output looks like

- A version line: `Experiment: summary-v7 | Notebook build: …`, then
  `Stage: pilot | Workspace: … | Model: Qwen/Qwen3.8-27B (vLLM 0.28.0)
  | Shared budget: 12 GPU hours`, then one `[n/6]` line per step. A CPU runtime
  stops at step 1 with the runtime-type fix, before Drive is mounted.
- Dependency installation is quiet; only pip errors and warnings are printed.
- Before weights, `STRUCTURED_OUTPUTS_OK` verifies decoder semantics and
  `DECODER_LATENCY_OK` reports the tokenizer-only mask timings. Failure stops setup.
- During inference, a line every 30 seconds: awaiting the first evaluation,
  then `successful evaluations: k/n`. Startup details remain in the phase's private
  server/runner logs. The first session downloads about 55 GB of weights before scoring.
- When context recovery is needed, `Context inventory: ... maximum request plus output:
  ... Context: 65536 -> ...` shows the measured requirement and selected window.
  A saved inventory is reused before model startup; a new token-only context failure
  permits one automatic pilot restart within the remaining allocation.
- `Finished: executed | Results: …` with the AUROC per condition, then the runtime
  disconnects.
- On failure: `Stopped: <ErrorClass>: <first message line>`, recognized hints
  (dependency mismatch, sampler initialization, GPU memory, context limit, uncertain time) and the private
  diagnostic path. Validation field details and log excerpts stay on Drive; the
  notebook output never repeats benchmark text.
- An incomplete phase prints the runner exit code, `successful evaluations: k/n`,
  counts per evaluation status and the worker's final handled error line before a
  partial report is saved.

| Stage | Work performed | What happens next |
| --- | --- | --- |
| `pilot` | Three development pairs, all four conditions, report | Inspect feasibility and failure rates |
| `development` | Complete/reuse the pilot, then all eight development pairs, reports | Review full development before test |
| `test` | Validate reviewed development, freeze locally, evaluate 27 held-out pairs, report | Held-out results stay separate |

An incomplete pilot stops automatic progression to development. Selecting test
requires **DEVELOPMENT_REVIEWED**, meaning the author has reviewed the completed
development results and authorizes the local protocol commit/tag and test run.
This is a later explicit run; nothing automatically advances from development to
test. The existing freeze validator still requires complete successful development,
matching model/runtime/config/prompts/code/data and every planned evaluation unit.
No public push occurs.

## Results and reconnects

The default private folder is `My Drive/agent-monitor-context-audit-private`.
The final cell prints full paths. For each phase:

- `numeric-results/summary-v7/<phase>/public_scores.csv`: sanitized numeric rows.
- `numeric-results/summary-v7/<phase>/reproduced/findings.md`: report.
- `numeric-results/summary-v7/<phase>/reproduced/metrics.json` and `figures/`: metrics and charts.
- `runs-private/notebook-status/summary-v7/latest.json`: workflow status and output locations.
- `runs-private/notebook-status/summary-v7/last-error.log`: last full private diagnostic.
- `runs-private/qwen-<phase>-summary-v7-ctx196608/gpu_sessions/server_logs/`
  and `runner_logs/`: engine/worker logs when inheriting the saved 196,608-token window.

After a context adjustment, active run directories have a `-ctx<tokens>` suffix.
The original attempt remains intact. The suffix reflects the actual saved window;
a workspace without a prior context choice initially has no context suffix.
Earlier legacy, `summary-v2`, `summary-v3`, `summary-v4`, `summary-v5` and
`summary-v6` results and source workspaces stay unchanged. The printed diagnostic paths identify the
selected version.

`versions/summary-v7/` contains this version's configuration, context selection,
source pin, durable Git metadata, source receipts and eventual frozen snapshot.
First setup inherits the newest valid parent source pin and context selection from
`summary-v6`, then `summary-v5`, `summary-v4`, `summary-v3`, `summary-v2` and legacy.
Reconnects retain the new version's own copies. Source files and Git metadata are isolated from earlier
versions.
The parent `configuration/model-pin.json`,
`data-private/` and `runs-private/gpu_budget/` remain shared; versioning does not
copy or reset model pins, dataset IDs, split assignments or GPU accounting.

Reconnect a GPU, keep the same workspace name, select the next stage and use
**Run all** again. Successful completed phases are checked against their recorded
signature and score checksum before being reused without loading the model.
Partial runs use the existing incremental cache and retry policy. Export and
analysis use bounded subprocesses (up to 120 seconds per phase); a timeout preserves
scores and logs and releases the runtime. Reports can also be reproduced later on
CPU using notebook 02 or the `export` / `analyze` CLI.

## Twelve cumulative GPU hours

The author authorized 12 GPU hours and no USD limit. The same ledger covers pilot,
development, test and resumptions. The notebook restores its immutable cap and
initial debit automatically, including the previously recorded 1,800 seconds.
There is no need to reenter 0.5 hours or maintain the old collection of flags.

**PRIOR_GPU_MINUTES** applies only when creating a brand-new budget. Enter allocation
already spent before starting the workflow. For an existing budget its saved initial
debit wins. A new folder is not authorization for another 12 hours: carry all prior
use into any explicitly exploratory workspace. Time before pressing Run all or
outside this workflow cannot be inferred; additional observed periods still use
`record_external_gpu_time` with a unique ID and confirmed duration.

Within a run, the notebook measures setup, acquisition and report overhead and
charges it to the shared ledger, subtracting time already charged by the managed
inference supervisor to avoid double counting. Each new model session is capped
by the remaining allowance and leaves 150 seconds for bounded reports and cleanup.
A notebook-level timer requests runtime release before the remaining allocation
expires. The core supervisor independently watches its own process groups.
These controls cannot certify Colab's billing cutoff or observe allocation outside
the workflow. Platform availability and disconnection behavior remain Colab's.

A known setup failure saves measured overhead for accounting on the next successful
setup. A kernel/VM lost without confirmed end time leaves a private unresolved
`runs-private/notebook-sessions` receipt and stops automatic resumption; review it
alongside any reserved core GPU session. Neither is silently settled to zero.
Unknown dollar costs remain null throughout.

## Matching source without manual patches

The notebook contains a compressed, checksummed copy of public runtime source,
prompts and the amended plan, decision record and operating guide, generated by
`scripts/build_qwen_notebook.py`. It contains no benchmark
content, model weights, credentials or private results. An initial repository clone
supplies Git provenance; its exact commit stays pinned on Drive. The embedded copy
repairs recognized old public-source versions without requiring a new branch push
or a separate patch upload. Changed originals are backed up privately.

All paths and hashes are checked before replacing files. Unknown local edits,
source symlinks, incompatible recorded scientific code hashes within the selected
version and differing frozen source are rejected. Only explicit versioned paths
scope the recorded-run guard to that version; the legacy path still protects
all recorded runs. Source updates never reset datasets, budget ledgers or results.
A frozen source snapshot takes precedence. Existing bundle-based workspaces remain
readable, but a new guided run only needs this notebook.

After changing public runtime code or either bootstrap script, rebuild with:

```bash
uv run python scripts/build_qwen_notebook.py
```

Keep committed notebook outputs empty. Setup records the embedded snapshot hash,
bootstrap hash, Git commit, exact model revision, installed packages and import checks
in private Drive provenance. A source refresh is not evidence of a successful model run.

## Automatic dependency and acquisition checks

The earlier startup failure was Torch CUDA 13.0 versus TorchAudio CUDA 12.8,
before model loading. For vLLM 0.28.0 / Torch 2.13.0 / CUDA 13.0, setup installs
`torchaudio==2.11.0+cu130` from the official PyTorch wheel index with `--no-deps`.
It preserves the pinned Torch and vLLM versions, then imports TorchAudio and the
vLLM server module in a fresh subprocess without launching a model. Other ABI
failures stop explicitly; no speculative library upgrade loop is performed.

Sources: [vLLM CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.28.0/requirements/cuda.txt),
[TorchAudio compatibility](https://docs.pytorch.org/audio/stable/installation.html),
[official CUDA 13 wheel index](https://download.pytorch.org/whl/cu130/torchaudio/).

Acquisition ignores filesystem-only executable-bit drift with command-scoped
`core.fileMode=false`. Pinned upstream HEAD/origin, tracked content, decryptor bytes
and decrypted payloads are still verified. Existing opaque IDs and family splits
are retained; the notebook never executes transcript commands.

## FlashInfer sampler startup on Blackwell

The subsequent private log confirms all checkpoint shards loaded, using 50.22 GiB,
before startup failed in the FlashInfer top-k/top-p sampler's architecture check
during dummy profiling. The message `FlashInfer requires GPUs with sm75 or higher`
does not establish that the reported SM120 GPU is unsupported or out of memory.
The log does not explain why FlashInfer selected an ineligible compilation target.

The managed server now sets `VLLM_USE_FLASHINFER_SAMPLER=0` before importing vLLM,
selecting its native PyTorch sampler for every stage. This override is fixed by
the source snapshot and recorded in private GPU session metadata; inherited values
cannot reenable FlashInfer sampling. Model/runtime pins, attention selection, BF16,
context length and sampling parameters remain unchanged. This is a supported switch
in [vLLM 0.28.0's sampler](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/sample/ops/topk_topp_sampler.py)
and [environment settings](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/envs.py).

Upload the rebuilt notebook, keep **STAGE = pilot** and the existing Drive workspace,
confirm the form and choose **Run all**. No additional variable is required. The
embedded update preserves the existing dataset and cumulative budget, including
failed sessions. It can refresh a workspace that only reached server startup;
incompatible recorded scientific runs or a frozen protocol still block source changes.
Offline tests verify launch configuration and notebook packaging. A fresh authorized
GPU pilot must still confirm startup and inference end to end.

## Inference contract

### Automatic recovery from a context-only pilot failure

The reported pilot reached the worker successfully but stopped before scoring because
four full requests exceeded the configured 65,536-token window. The existing preflight
counts all eligible transcripts, including test inputs for tokenization only. The
notebook now reads those saved counts before launching the next server. A complete,
matching inventory determines the largest prompt plus its configured output allowance;
the notebook adds up to 1,024 tokens of headroom and rounds up in 32,768-token steps,
within the model's native 262,144-token limit. No RoPE extension is used. The native
limit is recorded in the [pinned model configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json).

Recovery requires matching source, prompts, configuration and dataset provenance,
and no generation attempts, cached representations, request reservations, scores or
completion records in any phase of the selected version. A frozen protocol blocks it. Counts alone can
select context; evaluation outcomes cannot. If the inventory exceeds the native
limit, the workflow stops for scope/model review without exclusions or truncation.

The choice is saved to the selected source workspace's
`configuration/context/selection.json`, with an adjustment
history, the source preflight checksum and previous run identity. New phase configs
and run directories use `-ctx<tokens>` names; the original configs, token counts,
server logs and time receipts are retained. All subsequent stages, freeze checks,
exports and diagnostics use the same selected context. A new workspace that first
discovers this issue during its pilot can restart once automatically after the
server shuts down and the remaining time is checked. Later inference failures do
not trigger context tuning. The endpoint repeats exact token validation before
generation. Token feasibility alone does not guarantee hardware memory fit; any
new context choice still needs an authorized GPU pilot.

To resume the reported failure, upload the rebuilt notebook, keep the current Drive
folder and `STAGE = pilot`, confirm the form, then select **Run all**. There is no
context variable to edit and no configuration file to delete. For `summary-v7`,
the newest valid existing selection is inherited from summary-v6, then summary-v5,
summary-v4, summary-v3, summary-v2 and legacy; it is not recomputed from earlier generation failures. The larger
output allowances remain subject to exact request-token preflight validation.

### Model, hardware and sampling

Hardware eligibility requires one NVIDIA GPU with at least 75,000 MiB visible
VRAM and compute capability >=8.0 for native BF16. The session records the actual
GPU name, memory, driver and compute capability. This admits the author's RTX
PRO 6000 Blackwell Server Edition (97,887 MiB, SM120) as well as the original H100.
Eligibility does not validate all installed vLLM kernels or guarantee model fit.
Keep the same hardware across the study when possible and report any changes;
latency and GPU cost cannot be compared as if hardware were identical.

The author reported PyTorch 2.13.0+cu130, CUDA 13.0 and a passing small BF16 matrix
multiplication on SM120. That diagnostic checked basic PyTorch execution only;
later private pilot records reached model generation, including the 23-of-24
summary-v3 pilot and the complete 24-of-24 v6 pilot. V6 full development completed
63 of 64 evaluations. The revised v7 decoding contract still needs its own GPU
pilot. Keep vLLM's automatic attention selection and the existing BF16 configuration.

The only Qwen model is `Qwen/Qwen3.8-27B`, shared by independent monitor and
summarizer requests. vLLM is pinned to 0.28.0; the exact model/tokenizer SHA and
versions of vLLM, Torch, Transformers, Tokenizers, Triton and Safetensors are
preserved and checked across runtimes. The default is BF16, tensor parallelism 1,
one sequence, an initial 65,536-token context with the pre-scoring recovery above,
eager execution, chunked prefill,
text-only loading and no prefix cache or speculative decoding. Record the selected
context and observed GPU behavior for each version. Full is never truncated.

Body budgets use `/tokenize` without special tokens. Full request counts use the
same chat template, messages and generation marker as inference. Returned prompt
usage must agree with the count; discrepancies fail validation. Both
`enable_thinking` and `preserve_thinking` are false. Reasoning and tool-call outputs
are rejected and not retained. No execution tools are supplied to the models.

Non-thinking sampling starts at temperature 0.7, top-p 0.8, top-k 20 and presence
penalty 1.5. Seeds are derived from the configured seed and repeat/attempt identity.
All settings enter the cache/protocol signature. Fixed seeds do not guarantee
bitwise reproducibility across hardware or library changes. One realization of
this pipeline still does not establish monitor calibration or general safety.

## Interruptions and costs

The supervisor owns the loopback server and runner subprocess groups. An independent
watchdog terminates those groups at the reserved session deadline or if the
supervisor dies. Normal teardown also kills descendants after their leader exits.
Per-request cache records distinguish completed requests, retries and intentional
repetitions; completed numeric rows survive repeated interruptions.

Each managed GPU session reserves its maximum GPU seconds before process launch.
Known teardown settles elapsed time. A lost receipt remains conservatively reserved;
use `context_audit.colab.reconcile_gpu_session` with confirmed elapsed-time evidence
before resuming. Never set an uncertain session to zero merely to resume. An uncertain
individual model request is not automatically sent again.

The shared receipt records used, reserved and remaining seconds even when no
dollar rate is provided. Reconciliation obtains the budget from the original
session metadata and does not require a dollar cap. It changes accounting only;
generation remains a separate opt-in.

When an effective rate is supplied, `summary_cost_usd` and `monitor_cost_usd`
attribute request duration, including
attempts. `infrastructure_cost_usd` allocates the remaining cumulative managed
session cost equally across observed evaluation units, including model loading,
tokenization, idle gaps and interrupted work. Their sum is `cost_usd`. This
allocation is a reporting convention, not a measured causal per-condition GPU
cost. Partial runs retain explicit missing-unit coverage; entirely missing rows
cannot represent session expenditure, which is also recorded separately in the
exported manifest. Conservative request reservations can make reported totals
upper bounds above measured session time. Reanalysis is API-free but a restarted
GPU session has its own startup cost.

Without a rate, monetary costs remain unavailable in caches, numeric exports and
reports. Request latency and GPU-time receipts still record resource usage. Do not
interpret unavailable monetary costs as free inference or infer a dollar-price
comparison from them.

## Sources and verification limits

- [Official Qwen model card](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Qwen vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- [Pinned vLLM server options](https://docs.vllm.ai/en/v0.28.0/cli/serve/)
- [Colab resource and session limits](https://research.google.com/colaboratory/faq.html)

CPU tests exercise synthetic HTTP, process, persistence, protocol and notebook
boundaries. They do not validate a loaded Qwen checkpoint, GPU memory fit,
throughput, estimated price, data-use terms or empirical monitoring performance.
