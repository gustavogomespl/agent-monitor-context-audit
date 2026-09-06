# Guided Qwen Colab Implementation Plan

**Goal:** Run the selected scientific stage from one Colab form, including setup,
official acquisition, supervised inference, saved reports and GPU release.

**Architecture:** Keep scientific runners unchanged. A standard-library bootstrap
embedded in the notebook prepares the matching public runtime. A guided workflow
calls the existing acquisition, configuration, budget, freeze and reporting APIs.
The notebook ships its public source snapshot so a stale Drive checkout does not
require a separate patch upload. Existing recorded methods and private data win
over incompatible updates.

**Tech stack:** Python, native Colab forms, existing vLLM supervisor and Drive storage.
**Spec:** User request in this task; `research_plan.md`; `docs/qwen_colab.md`.

## Constraints

- Explicit start, data-use and rubric confirmations; reviewed development before test.
- Twelve cumulative GPU hours, existing initial debit restored, unknown USD left null.
- Preserve model/runtime pins, datasets, opaque IDs, cache and frozen protocols.
- Benchmark content stays private; no notebook outputs, public push or local commit here.
- No empirical claim from offline tests.

## Implementation

- [x] Write synthetic workflow tests for successful stage ordering, review gates,
  failures, reports, cleanup and reuse of the existing initial debit.
- [x] Extract existing notebook helpers into `scripts/colab_bootstrap.py`; add the
  guided coordinator and concise progress/status output. Use the existing runtime
  APIs, including `initialize_gpu_budget` and `run_colab_experiment`.
- [x] Write snapshot tests; implement a checked public-source payload with complete
  prevalidation, preservation of local edits and recorded scientific code hashes.
- [x] Add `scripts/build_qwen_notebook.py` to generate an empty-output notebook with
  one primary form, optional advanced settings, hidden implementation and one run cell.
- [x] Automate setup/import repair, acquisition, phase execution, bounded numeric
  report subprocesses and GPU release, including setup/error paths. Account measured
  notebook overhead without resetting the existing budget.
- [x] Update notebook integration tests and instructions around Run All and stage
  selection. Keep the existing source-pin and CUDA dependency regression coverage.
- [x] Run targeted tests, full `uv run pytest`, `uv run ruff check .`, notebook
  execution in the offline default state, public scan and diff checks.

The pilot is a complete first run. Development can run the pilot then all development
pairs; test is a later explicit selection after review and performs the local freeze
before generation. Failures stop progression and retain null scores and private logs.

Validation: 297 synthetic tests passed; Ruff, the public-content scan and diff checks passed. The guided notebook also executed with its inert defaults in a real local Jupyter kernel. No GPU model inference was performed.
