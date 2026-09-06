# Qwen3.8-27B on Colab

Open `notebooks/03_qwen_colab.ipynb` through Colab's **File → Upload notebook**.
The default source is this GitHub repository's `pilot` branch. Set `BRANCH` to
another published branch when needed; `CODE_REF` optionally selects an exact
40-character commit. Default Run All performs no
installation, download, Drive mounting, data acquisition or model generation.

## Running the notebook

1. In a preparation runtime, set `BRANCH="pilot"`, `RUN_SETUP=True` and allow Drive
   mounting. Setup clones `https://github.com/gustavogomespl/agent-monitor-context-audit.git`,
   fetches the selected branch, checks out its exact commit and saves it in
   `configuration/code-pin.json` on Drive. It also installs dependencies and saves
   the model/runtime pins; it never starts a model. No ZIP is required.
2. Set `RUN_ACQUIRE=True` once. The official pinned benchmark is acquired and
   inventoried privately. Existing records retain their opaque IDs and split on
   reconnect. Review the family grouping and fixed rubric before generation.
3. Select an eligible GPU runtime (for example H100 or RTX PRO 6000 Blackwell)
   and rerun setup. Keep `MAX_GPU_HOURS=12.0`, `MAX_COST_USD=None` and
   `GPU_HOURLY_RATE_USD=None`. Before the first live run, set
   `GPU_ALREADY_USED_HOURS` to GPU allocation already spent on setup or idle.
   Complete the data-use/rubric confirmation flags after review. Use
   `PHASE="pilot"`, `RUN_LIVE=True`. All eligible full
   requests are counted before generation; an oversized input stops the run.
4. Inspect private pilot results, failure rates, observed summary lengths and
   actual context/memory behavior. Complete `PHASE="development"` only after
   choosing final methods. The pilot evaluates three pairs; full development
   evaluates every development pair. Their distinct run directories share the
   same cumulative GPU-hour allowance.
5. With generation disabled, explicitly enable `RUN_FREEZE` and
   `REVIEWED_FREEZE` after reviewing complete development evidence. The notebook
   creates a local protocol commit/tag and durable source snapshot; it never
   pushes. Run test separately with `PHASE="test"` and `RUN_TEST=True`; it consumes
   the remaining shared GPU-hour allowance.
6. Numeric exports and offline reports persist on Drive. `RUN_ANALYSIS=True`
   reproduces them without generation. Notebook 02 uses the same frozen bootstrap
   parameters. Keep notebook outputs empty when committing or sharing.

The author authorized **12 cumulative GPU hours without a USD cap** on 2026-09-06.
The notebook's default session is at most one hour, reduced when the shared
remaining allowance is smaller. The time ledger persists on Drive and covers
pilot, development, test and resumptions within the same private workspace.
Its initial cap and prior-use debit are immutable; keep `GPU_ALREADY_USED_HOURS`
unchanged when resuming. A new run directory does not replenish that allowance.
For setup/idle allocation on later GPU runtimes, set `GPU_ADDITIONAL_USED_HOURS`
and a stable unique `GPU_ADDITIONAL_USAGE_ID` for the observed period before live
execution. Repeating the same ID and duration is idempotent; a new period needs a
new ID. This adds explicit outside-block usage to the shared ledger.

The notebook normally disconnects Colab immediately after managed execution;
private numeric rows and receipts have already been saved. Run exports and reports
later on CPU with `RUN_ANALYSIS=True`. The supervisor
measures managed process startup through teardown plus the declared prior use;
it cannot observe later setup/idle allocation outside its block or guarantee the
platform's billing cutoff. Declare those extra periods and release
the runtime promptly. A supplied dollar rate is optional and remains an estimate,
not a Colab invoice. Without it, USD fields are null rather than a fabricated zero.
GPU availability and platform session limits remain controlled by Colab.

## Source versions and reconnects

`BRANCH` selects the published source on the first setup. Later sessions retain
the saved commit even if the remote branch advances; they do not pull new code
into an existing experiment. Setup prints the selected commit and Python version.
Changing the source selection requires a new `DRIVE_ROOT` and fresh `/content`
runtime so existing records retain their provenance. A reviewed local freeze
takes precedence and is restored from its durable source snapshot.

For an unpublished working tree, set `PROJECT_ZIP` to the uploaded companion
`dist/qwen-colab-bundle.zip`. Alternatively clear `REPO_URL` to use the upload
picker. An existing saved ZIP workspace continues using that source. The ZIP is
optional; the default Git workflow needs only the uploaded notebook and access
to the selected repository branch.

## Acquisition on mounted filesystems

Some filesystems change the executable permission bit of checked-out files.
Source verification ignores that worktree bit with command-scoped
`core.fileMode=false`, while retaining the pinned HEAD, official origin, tracked
content comparison, decryptor byte comparison and decrypted-payload verification.
The benchmark is read as data; its transcript commands are never executed.

For an older pinned notebook that reports `Tracked official source files changed
after acquisition`, compare `git diff --quiet HEAD --` with `core.fileMode=true`
and `core.fileMode=false` in `data/private/upstream`. Exit codes 1 and 0,
respectively, identify permission-only differences. In that case, set
`git config --local core.fileMode false` in that upstream checkout and rerun the
acquisition cell. If the second comparison also fails, preserve the source and
investigate the content or Git error; do not reset files or disable verification.

## Inference contract

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
one sequence, 65,536 total context tokens, eager execution, chunked prefill,
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
the notebook provides a separate, explicit reconciliation phase using elapsed-time
evidence. Never set an uncertain session to zero merely to resume. An uncertain
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
