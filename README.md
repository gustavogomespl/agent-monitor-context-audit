# Agent Monitor Context Audit

A small retrospective experiment on whether compressing an agent's history
changes an external monitor's ability to distinguish problematic transcripts
from paired benign ones. It compares full normalized history, deterministic
head/tail text, a prose summary and a structured summary under a common
per-example token ceiling.

**Research status: no real API experiment has been executed.** Synthetic fixtures
demonstrate the engineering pipeline; they are never empirical results. The
directional hypothesis that structured summaries improve AUROC is untested.
The full scope is in the user-supplied [research plan](research_plan.md), and
current empirical status is in [findings](reports/findings.md).

## Qwen on a Colab H100

The Qwen workflow is in [03_qwen_colab.ipynb](notebooks/03_qwen_colab.ipynb), with
[setup, cost and recovery instructions](docs/qwen_colab.md). Upload the notebook
to Colab, select `BRANCH` (default `pilot`) and enable `RUN_SETUP`. Setup clones
this repository and saves the branch's exact commit for subsequent sessions.
The companion `dist/qwen-colab-bundle.zip` remains an optional source transfer.
It uses one pinned `Qwen/Qwen3.8-27B` in BF16 for
both roles, exact tokenizer counts, thinking disabled and persistent private
Drive storage. Each phase requires explicit opt-in, a positive GPU rate/cap and
data-use/rubric confirmation. Default Run All is inert. Actual H100 and model
performance validation remains pending a real pilot.

The local CPU environment remains lightweight; vLLM/CUDA are installed only in
the explicit Colab setup. The Anthropic workflow below remains available as a
separate backend; one run never mixes providers.

## Local setup and offline checks

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required. No GPU, agent
framework, web server or paid account is needed for the fixture demonstration.

```bash
uv sync --group dev
uv run context-audit demo
uv run pytest
uv run ruff check .
```

The demo uses independent `synthetic_fixture` examples and makes no API calls.
It must not write empirical performance into `results/public_scores.csv`.
CI uses only these independent fixtures, without real transcripts or API keys.

## Real benchmark workflow

```bash
uv sync --group dev --extra data
uv run context-audit acquire
uv run context-audit inventory
```

Acquisition follows the official
[SLEIGHT-Bench access instructions](https://github.com/safety-research/sleight-bench)
and records an exact source commit. Source content, annotations, opaque-ID
mappings and derived text stay under gitignored private directories. The
inventory resolves whole scenarios and sessions, excludes multissession pairs,
and records grouped development/test splits. Inspect the generated sanitized
manifest rather than assuming that a file is an independent case.

The pinned local inventory contains **35 eligible pairs (70 transcripts)** in
33 families: 8 development pairs and 27 test pairs, with seed `20260905`.
This differs from the README-based expectation in the original plan. The source
has 44 directories; exclusions cover six directories forming three multissession
cases, one unpaired case and two malformed pairs. See the
[inventory](data/manifests/inventory.json) and [opaque split](data/manifests/split.csv).
Automated family grouping still needs author review. Model-specific token counts
and full-context feasibility remain unverified until model selection and
authorized provider counting.

Use `--help` on each command for paths and configuration options. The
implementation is a local offline parser and API evaluation client: **it never
executes transcript commands or exposes tools to the models.** All conditions
exclude thinking. The first literal user request is a shared uncompressed
header; a tool result serialized with `role=user` is not human authorization.

Before a real pilot, copy `.env.example` to `.env` locally, supply the API key,
configure explicit monitor/summarizer model IDs and verified context windows in
`configs/pilot.yaml`, and copy `configs/prices.example.yaml` to the gitignored
`configs/prices.local.yaml`. Fill it from
[official pricing](https://platform.claude.com/docs/en/about-claude/pricing),
including its date, units and model-specific rates.
The user must confirm the account's data use is compatible with the benchmark
restrictions, and explicitly opt into a financial cap and live execution.
Missing data/access causes an informative error, never fixture substitution.
Record completed data-use and rubric reviews using `data_use_confirmed` and
`rubric_reviewed` in the configuration; their initial `false` values are intentional.

```bash
# Requires configured models, reviewed data use/rubric, key and a pilot cap.
uv run context-audit run --config configs/pilot.yaml --max-cost-usd PILOT_CAP --live

# Prepare the full development run after pilot revisions are finished.
cp configs/pilot.yaml configs/development.yaml
```

In `configs/development.yaml`, set `pilot_pairs: null` and
`run_dir: runs/private/development` to include all eight development pairs.
Keep final models, prompts and generation/budget settings aligned with
`configs/main.yaml`. Then, with a separately authorized development cap:

```bash
uv run context-audit run --config configs/development.yaml --max-cost-usd DEV_CAP --live

# After all development pairs succeed and the author completes review.
uv run context-audit freeze --config configs/main.yaml \
  --development-run runs/private/development --reviewed

# Only after protocol-v1 is fixed and a test-run budget is authorized.
uv run context-audit run --config configs/main.yaml --max-cost-usd TEST_CAP --live

# Local sanitized export and offline analysis; these make no API calls.
uv run context-audit export --run-dir runs/private/main --output results
uv run context-audit analyze --scores results/public_scores.csv \
  --manifest results/run_manifest.json
uv run context-audit scan-public
```

Replace `PILOT_CAP`, `DEV_CAP` and `TEST_CAP` with explicitly authorized amounts;
there is no default spend. Caps are cumulative **per run directory**, including
its retries and resumes, not a project-wide account limit. Allocate any total
project budget across stages. Conservative pre-request reservations use the
model context-window bound and maximum output, so a small remaining cap can stop
calls even if their likely bill would fit. Unknown charges remain labeled upper
bounds until reconciled.

The freeze command requires a completed full-development run with matching
methods. It prepares a manifest; the user must then review and commit the
protocol/code/config/prompts and tag that HEAD `protocol-v1` before the test
command can run. No commit, push or publication is implied here. Prices and
model capabilities are verified at pilot time. The checked-in lockfile currently
resolves the official `anthropic` SDK to 1.4.0; the run records its actual version.

## Analysis and notebooks

- [Dataset and pilot notebook](notebooks/01_dataset_and_pilot.ipynb) explains the
  question, inventory and representations. Real execution requires `RUN_LIVE=true`
  and explicit configuration/budget; opening it never triggers generation.
- [Reproduction notebook](notebooks/02_reproduce_results.ipynb) uses sanitized
  real numeric results only, recomputing metrics, paired bootstrap and figures
  on CPU without API calls. When results are absent it says the experiment was
  not executed. Committed notebook outputs stay empty.

The primary outcome is `AUROC(structured_summary) - AUROC(free_summary)` on the
same complete pairs with valid scores in all conditions. Family bootstrap keeps
pairs and conditions together. Failures have null scores and route to review;
coverage and end-to-end failure effects are reported. Cost includes summaries,
monitoring and retries. No production safety, real-time prevention, calibrated
probability or reliable 1% FPR claim follows from this small study.
The sanitized run manifest supplies planned coverage denominators. Analysis
without verified planned units cannot pass the primary coverage quality gate.

## Documentation and storage

Read [methodology](docs/methodology.md), [related work](docs/related_work.md),
[verified references](docs/references.md), [data handling](docs/data_handling.md)
and [decisions / AI assistance](docs/decisions.md). `data/private/`,
`runs/private/`, `.env` and text-bearing caches stay out of Git. Public artifacts
are generic code/prompts, sanitized manifests, numeric scores with opaque IDs,
metrics, figures and independent fixtures. Source canaries remain with private
derived artifacts outside model-visible inputs. A content scan and human review
precede publication; publishing requires explicit authorization.

Project code retains the existing [Apache-2.0 license](LICENSE). The separately
acquired upstream source carries its own
[MIT license](https://github.com/safety-research/sleight-bench/blob/main/LICENSE)
and anti-contamination notices; no benchmark collection is redistributed here.
