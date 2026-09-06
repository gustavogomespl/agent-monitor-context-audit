# Qwen3.8-27B on Colab

Open `notebooks/03_qwen_colab.ipynb` through Colab's **File → Upload notebook**.
The companion `dist/qwen-colab-bundle.zip` transfers the current public source and
Git provenance without publishing this checkout. Default Run All performs no
installation, download, Drive mounting, data acquisition or model generation.

## Running the notebook

1. In a preparation runtime, set `RUN_SETUP=True`. Upload the companion bundle
   when prompted and allow Drive mounting. Alternatively provide a repository URL
   and exact commit that already contains this implementation. Setup preserves a
   durable model SHA and inference package versions; it never starts a model.
2. Set `RUN_ACQUIRE=True` once. The official pinned benchmark is acquired and
   inventoried privately. Existing records retain their opaque IDs and split on
   reconnect. Review the family grouping and fixed rubric before generation.
3. Select an H100 runtime and rerun setup. Fill your effective
   `GPU_HOURLY_RATE_USD`, cumulative per-phase `MAX_COST_USD`, and data-use/rubric
   confirmation flags. Use `PHASE="pilot"`, `RUN_LIVE=True`. All eligible full
   requests are counted before generation; an oversized input stops the run.
4. Inspect private pilot results, failure rates, observed summary lengths and
   actual context/memory behavior. Complete `PHASE="development"` only after
   choosing final methods. The pilot evaluates three pairs; full development
   evaluates every development pair. They have separate costs and run directories.
5. With generation disabled, explicitly enable `RUN_FREEZE` and
   `REVIEWED_FREEZE` after reviewing complete development evidence. The notebook
   creates a local protocol commit/tag and durable source snapshot; it never
   pushes. Run test separately with `PHASE="test"`, `RUN_TEST=True` and its cap.
6. Numeric exports and offline reports persist on Drive. `RUN_ANALYSIS=True`
   reproduces them without generation. Notebook 02 uses the same frozen bootstrap
   parameters. Keep notebook outputs empty when committing or sharing.

The notebook normally disconnects the Colab runtime after durable export.
Installation and GPU allocation while reading/editing the notebook are outside
the managed execution cap. A supplied dollar rate is an estimate, not a Colab
invoice. GPU availability and platform session limits remain controlled by Colab.

## Inference contract

The only Qwen model is `Qwen/Qwen3.8-27B`, shared by independent monitor and
summarizer requests. vLLM is pinned to 0.28.0; the exact model/tokenizer SHA and
versions of vLLM, Torch, Transformers, Tokenizers, Triton and Safetensors are
preserved and checked across runtimes. The default is BF16, tensor parallelism 1,
one sequence, 65,536 total context tokens, eager execution, chunked prefill,
text-only loading and no prefix cache or speculative decoding. Context capacity
is a pilot setting, not a demonstrated H100 result. Full is never truncated.

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

Each managed GPU session reserves its maximum time cost before process launch.
Known teardown settles elapsed time. A lost receipt remains conservatively reserved;
the notebook provides a separate, explicit reconciliation phase using elapsed-time
evidence. Never set an uncertain session to zero merely to resume. An uncertain
individual model request is not automatically sent again.

`summary_cost_usd` and `monitor_cost_usd` attribute request duration, including
attempts. `infrastructure_cost_usd` allocates the remaining cumulative managed
session cost equally across observed evaluation units, including model loading,
tokenization, idle gaps and interrupted work. Their sum is `cost_usd`. This
allocation is a reporting convention, not a measured causal per-condition GPU
cost. Partial runs retain explicit missing-unit coverage; entirely missing rows
cannot represent session expenditure, which is also recorded separately in the
exported manifest. Conservative request reservations can make reported totals
upper bounds above measured session time. Reanalysis is API-free but a restarted
GPU session has its own startup cost.

## Sources and verification limits

- [Official Qwen model card](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Qwen vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- [Pinned vLLM server options](https://docs.vllm.ai/en/v0.28.0/cli/serve/)
- [Colab resource and session limits](https://research.google.com/colaboratory/faq.html)

CPU tests exercise synthetic HTTP, process, persistence, protocol and notebook
boundaries. They do not validate a loaded Qwen checkpoint, H100 memory fit,
throughput, estimated price, data-use terms or empirical monitoring performance.
