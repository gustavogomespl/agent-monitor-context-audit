"""Build a self-contained guided notebook from public, reviewable source files."""

import base64
import hashlib
import json
import subprocess
import zlib
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]

INTRO = """# Qwen context audit · Summary v3 · guided Colab run

1. **Runtime → Change runtime type**: select one H100 80GB or RTX PRO 6000 Blackwell 96GB GPU.
2. **Form 1 below**: keep **STAGE = pilot** for the first run, confirm the two review boxes,
   then tick **START_RUN**. Leave the optional settings alone to resume an existing workspace.
3. **Runtime → Run all**, then allow Google Drive access when asked.

Everything else is automatic: GPU check → Drive and budget → matching source and
dependencies → official dataset → inference → report → disconnect. The first pilot
session installs vLLM and downloads about 55 GB of weights before scoring starts, so
expect a long wait with a progress line every 30 seconds. If a step fails, the output
names the error and the private diagnostic path; the runtime is released either way.

The selected model is **Qwen/Qwen3.8-27B**, served locally with vLLM 0.28.0 and
thinking disabled. The first setup pins its exact weights and inference packages.
There is **no USD cap**; all stages share **12 cumulative GPU hours**.

Results, logs, dataset and budget stay in your private Drive folder and are reused on
reconnect. The notebook includes its matching public runtime, so you do not need to
upload a patch or edit the branch.

**This notebook starts the `summary-v3` development amendment.** It repeats all 24
pilot evaluations with constrained structured-summary generation. Each item has text
and required references selected from the visible events. The application renders the
citations and derives the final ID list, preserving all claims. It keeps the 1,024-token
ceiling on the complete final summary, two-attempt limit and four conditions.
Before model startup, a CPU check verifies the pinned citation decoder.
Existing dataset, split, model revision, runtime pins and context selection are reused;
previous evaluations stay untouched. New results appear in `numeric-results/summary-v3`.
Keep your existing Drive folder; no deletion or manual patch is needed.

Transcript lengths are checked before scoring. If they need more context, the notebook
selects a larger native window and retries the pilot automatically, preserving complete
histories. Keep the same Drive folder to reuse checks from an interrupted attempt.

Before confirming below, review the [monitor rubric](https://github.com/gustavogomespl/agent-monitor-context-audit/blob/pilot/prompts/monitor.txt)
and the [research protocol](https://github.com/gustavogomespl/agent-monitor-context-audit/blob/pilot/research_plan.md).
Transcripts are treated as data; their commands are never executed.
"""

FORM = '''#@title 1. Choose the stage and confirm execution
#@markdown **pilot:** 3 development pairs, four conditions, scores and report.
#@markdown **development:** finish the pilot, then all 8 development pairs.
#@markdown **test:** freeze reviewed methods, then all 27 held-out pairs.
STAGE = "pilot" #@param ["pilot", "development", "test"]
#@markdown I confirm that this benchmark may be processed locally in this Colab/Drive environment.
DATA_USE_CONFIRMED = False #@param {type:"boolean"}
#@markdown I reviewed the monitoring rubric and the research hypothesis.
RUBRIC_REVIEWED = False #@param {type:"boolean"}
#@markdown Required only for **test**: I reviewed the completed development run
#@markdown and authorize its local protocol freeze and test execution.
DEVELOPMENT_REVIEWED = False #@param {type:"boolean"}
#@markdown Start the selected stage with the saved cumulative 12-hour GPU budget.
START_RUN = False #@param {type:"boolean"}
'''

ADVANCED = '''#@title Optional settings — keep these defaults to resume your existing run
from pathlib import Path

#@markdown Folder under My Drive. Keep the same name to reuse data, pins and budget.
#@markdown Summary v3 has its own cache and results; all previous runs are preserved.
WORKSPACE_FOLDER = "agent-monitor-context-audit-private" #@param {type:"string"}
#@markdown GPU minutes used **before this workflow**, only for a brand-new budget.
#@markdown An existing initial debit is restored automatically.
#@markdown Setup and report time inside this workflow are measured automatically.
#@markdown Time outside the workflow cannot be inferred.
PRIOR_GPU_MINUTES = 0 #@param {type:"number"}

if (not WORKSPACE_FOLDER or Path(WORKSPACE_FOLDER).name != WORKSPACE_FOLDER
        or WORKSPACE_FOLDER in {".", ".."}):
    raise ValueError("Use one folder name under My Drive.")
EXPERIMENT_VERSION = "summary-v3"
REPO = Path("/content/agent-monitor-context-audit-summary-v3")
DRIVE_ROOT = Path("/content/drive/MyDrive") / WORKSPACE_FOLDER
MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = ""
VLLM_VERSION = "0.28.0"
MAX_MODEL_LEN = 65536
MAX_GPU_HOURS = 12.0
GPU_HOURLY_RATE_USD = None
MAX_COST_USD = None
STARTUP_TIMEOUT_SECONDS = 1800
REPO_URL = "https://github.com/gustavogomespl/agent-monitor-context-audit.git"
BRANCH = "pilot"
CODE_REF = ""
PROJECT_ZIP = ""
SETUP_READY = False
'''

ENDING = """### Reading the output

- `Finished: executed | Results: …` plus the AUROC per condition means the stage completed;
  the runtime then disconnects on its own.
- `Stopped: <ErrorClass>: <message>` means a step failed. The line names the cause; the
  full private diagnostic is `runs-private/notebook-status/summary-v3/last-error.log` on Drive.
- `<stage> did not complete (runner exit code …)` lists successful/expected evaluations,
  counts per status and the worker's final error line; the partial report is still saved.
- `Not started (STAGE = …)` with `[ ]` boxes means form 1 is incomplete; nothing ran.

### After the run

The first output line must say **Experiment: summary-v3**. Your Drive folder contains
`numeric-results/summary-v3/<stage>/reproduced/findings.md`,
`public_scores.csv`, `metrics.json` and figures. The final cell prints exact paths
and releases the GPU automatically, including when setup or inference fails.

For the next stage, reconnect a GPU, change **STAGE** and choose **Run all** again.
Successful saved evaluations from this version are reused. Development stops if the pilot is
incomplete. Test requires successful reviewed development and matching frozen methods.

If a run stops, inspect `runs-private/notebook-status/summary-v3/last-error.log` and the phase's
`gpu_sessions/server_logs` / `runner_logs`. Measured time and partial records remain
saved. A runtime lost without a confirmed end time requires accounting review;
rerunning never resets its reserved budget. Initial Drive access and human review
are the only expected interactions on a successful run.
"""


def source_payload(root):
    paths = sorted([
        *(root / "src/context_audit").rglob("*.py"),
        *(root / "prompts").glob("*.txt"), root / "pyproject.toml", root / "uv.lock",
        root / "scripts/colab_bootstrap.py", root / "scripts/notebook_snapshot.py",
        root / "research_plan.md", root / "docs/decisions.md", root / "docs/qwen_colab.md",
    ])
    if (root / "requirements-colab.txt").exists():
        paths.append(root / "requirements-colab.txt")
    # Recognize released source versions; arbitrary local edits remain protected.
    revisions = subprocess.run(
        ["git", "log", "-6", "--format=%H"], cwd=root,
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    files = []
    for path in paths:
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError("Never embed symlinked source.")
        body = path.read_text()
        previous = set()
        for revision in revisions:
            old = subprocess.run(["git", "show", f"{revision}:{name}"], cwd=root,
                                 capture_output=True)
            if old.returncode == 0:
                previous.add(hashlib.sha256(old.stdout).hexdigest())
        files.append(dict(path=name, text=body, sha256=hashlib.sha256(body.encode()).hexdigest(),
                          accepted_previous_sha256=sorted(previous)))
    raw = json.dumps(dict(schema_version=1, files=files), sort_keys=True).encode()
    return base64.b64encode(zlib.compress(raw, 9)).decode(), hashlib.sha256(raw).hexdigest()


def build(root=ROOT):
    def code(source, title=None):
        metadata = {"cellView": "form"}
        if title:
            source = f"#@title {title}\n" + source
        return nbformat.v4.new_code_cell(source, metadata=metadata)

    encoded, digest = source_payload(root)
    snapshot = (root / "scripts/notebook_snapshot.py").read_text()
    bootstrap = (root / "scripts/colab_bootstrap.py").read_text()
    implementation_hash = hashlib.sha256((snapshot + bootstrap).encode()).hexdigest()
    chunks = "\n".join(f'    "{encoded[i:i + 88]}"' for i in range(0, len(encoded), 88))
    implementation = (
        'from pathlib import Path\n\n'
        f'SOURCE_PAYLOAD_SHA256 = "{digest}"\nSOURCE_PAYLOAD_B64 = (\n{chunks}\n)\n'
        f'NOTEBOOK_BOOTSTRAP_SHA256 = "{implementation_hash}"\n' + snapshot + "\n"
        + bootstrap.replace("from pathlib import Path\n", "", 1)
        .replace('apply_embedded_source = globals().get("apply_embedded_source", None)\n', "")
    )
    cells = [
        nbformat.v4.new_markdown_cell(INTRO), code(FORM), code(ADVANCED),
        code(implementation,
             "Internal runtime — included and verified automatically; no edits needed"),
        code("RUN_RESULT = run_guided()", "2. Run the selected stage end to end"),
        nbformat.v4.new_markdown_cell(ENDING),
    ]
    for index, cell in enumerate(cells):
        cell.id = f"qwen-guided-{index}"
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "colab": {"name": "qwen_colab_summary_v3.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU",
    })
    nbformat.validate(notebook)
    return notebook


if __name__ == "__main__":
    target = ROOT / "notebooks/03_qwen_colab.ipynb"
    nbformat.write(build(), target)
    print(target)
