# Qwen3.8-27B on Colab

## Run in four steps

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gustavogomespl/agent-monitor-context-audit/blob/pilot/notebooks/03_qwen_colab.ipynb)

1. Upload the supplied `dist/qwen_colab_summary_v2_<hash>.ipynb` with
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

## Development amendment: summary-v2

The author authorized a new development version after offline diagnosis of saved
pilot summary failures. Both summary prompts now target 60–80% of the same body
budget, leaving room for citations and formatting under the unchanged per-example
cap (at most 1,024 tokens). Both preserve the same evidence priorities. One allowed
regeneration restates all length, format, schema and citation requirements using
the original input; neither the rejected candidate nor validator error text is
fed back. The limit remains two attempts total. No validator is relaxed and no
oversized summary is silently shortened.

The hidden setting `EXPERIMENT_VERSION = "summary-v2"` selects an isolated source
checkout at `/content/agent-monitor-context-audit-summary-v2` and new run identities.
The pilot starts all 24 evaluation units afresh across the same three development
pairs and four conditions; it does not reuse the previous pilot's successful or
failed calls. The model/tokenizer/runtime pins, dataset, opaque IDs, family split,
selected context and cumulative GPU allowance are retained. Existing runs, caches,
configs and reports remain available for comparison. Later reconnects can resume
this version's own incremental records.

This amendment was motivated by development validation failures, not held-out
scores. A first version setup stops if the parent workspace is already frozen or
contains a test run. Existing data-use/rubric confirmations, explicit start and
development review before test still apply. The new prompts have only offline
synthetic validation until the authorized Colab pilot is executed; improved live
coverage or monitoring performance is not established.

## What the output looks like

- A header line: `Stage: pilot | Workspace: … | Model: Qwen/Qwen3.8-27B (vLLM 0.28.0)
  | Shared budget: 12 GPU hours`, then one `[n/6]` line per step. A CPU runtime
  stops at step 1 with the runtime-type fix, before Drive is mounted.
- Dependency installation is quiet; only pip errors and warnings are printed.
- During inference, a line every 30 seconds: model loading, then
  `successful evaluations: k/n`. The first session downloads about 55 GB of weights
  before scoring starts.
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

- `numeric-results/summary-v2/<phase>/public_scores.csv`: sanitized numeric rows.
- `numeric-results/summary-v2/<phase>/reproduced/findings.md`: report.
- `numeric-results/summary-v2/<phase>/reproduced/metrics.json` and `figures/`: metrics and charts.
- `runs-private/notebook-status/summary-v2/latest.json`: workflow status and output locations.
- `runs-private/notebook-status/summary-v2/last-error.log`: last full private diagnostic.
- `runs-private/qwen-<phase>-summary-v2-ctx196608/gpu_sessions/server_logs/`
  and `runner_logs/`: engine/worker logs when inheriting the saved 196,608-token window.

After a context adjustment, active run directories have a `-ctx<tokens>` suffix.
The original attempt remains intact. The suffix reflects the actual saved window;
a workspace without a prior context choice initially has no context suffix.
Earlier `numeric-results/<phase>` and `qwen-<phase>-ctx<tokens>` artifacts stay
unchanged. The printed diagnostic paths always point to the selected version.

`versions/summary-v2/` contains this version's configuration, context selection,
source pin, durable Git metadata, source receipts and eventual frozen snapshot.
On first setup it inherits the parent source pin, context selection and public
manifests. Reconnects retain its own copies. The parent `configuration/model-pin.json`,
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
version and differing frozen source are rejected. Only the explicit `summary-v2`
path scopes the recorded-run guard to that version; the legacy path still protects
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
generation. Hardware memory fit and long-context inference still need live validation.

To resume the reported failure, upload the rebuilt notebook, keep the current Drive
folder and `STAGE = pilot`, confirm the form, then select **Run all**. There is no
context variable to edit and no configuration file to delete. For `summary-v2`,
the existing parent selection is inherited before startup; it is not recomputed
from the earlier generation failures.

### Model, hardware and sampling

Hardware eligibility requires one NVIDIA GPU with at least 75,000 MiB visible
VRAM and compute capability >=8.0 for native BF16. The session records the actual
GPU name, memory, driver and compute capability. This admits the author's RTX
PRO 6000 Blackwell Server Edition (97,887 MiB, SM120) as well as the original H100.
Eligibility does not validate all installed vLLM kernels or guarantee model fit.
Keep the same hardware across the study when possible and report any changes;
latency and GPU cost cannot be compared as if hardware were identical.

The author reported PyTorch 2.13.0+cu130, CUDA 13.0 and a passing small BF16 matrix
multiplication on SM120. That checks basic PyTorch execution only; loading Qwen
and its hybrid attention kernels still requires the authorized pilot. Keep vLLM's
automatic attention selection and the existing BF16 configuration.

The only Qwen model is `Qwen/Qwen3.8-27B`, shared by independent monitor and
summarizer requests. vLLM is pinned to 0.28.0; the exact model/tokenizer SHA and
versions of vLLM, Torch, Transformers, Tokenizers, Triton and Safetensors are
preserved and checked across runtimes. The default is BF16, tensor parallelism 1,
one sequence, an initial 65,536-token context with the pre-scoring recovery above,
eager execution, chunked prefill,
text-only loading and no prefix cache or speculative decoding. Context capacity
is a pilot setting, not a demonstrated GPU result. Full is never truncated.

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
