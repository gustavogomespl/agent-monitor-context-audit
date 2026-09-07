"""Standard-library Colab bootstrap, embedded verbatim by build_qwen_notebook.py.

Importing this file defines helpers only. The notebook form supplies configuration.
Scientific generation and dataset operations stay in the existing package.
"""

from pathlib import Path

# Defaults are inert; preserve the settings and snapshot function supplied by the notebook.
START_RUN = globals().get("START_RUN", False)
STAGE = globals().get("STAGE", "pilot")
EXPERIMENT_VERSION = globals().get("EXPERIMENT_VERSION", "legacy")
DATA_USE_CONFIRMED = globals().get("DATA_USE_CONFIRMED", False)
RUBRIC_REVIEWED = globals().get("RUBRIC_REVIEWED", False)
DEVELOPMENT_REVIEWED = globals().get("DEVELOPMENT_REVIEWED", False)
PRIOR_GPU_MINUTES = globals().get("PRIOR_GPU_MINUTES", 0)
DRIVE_ROOT = globals().get(
    "DRIVE_ROOT", Path("/content/drive/MyDrive/agent-monitor-context-audit-private"),
)
REPO = globals().get("REPO", Path("/content/agent-monitor-context-audit"))
MODEL_ID = globals().get("MODEL_ID", "Qwen/Qwen3.8-27B")
MODEL_REVISION = globals().get("MODEL_REVISION", "")
VLLM_VERSION = globals().get("VLLM_VERSION", "0.28.0")
MAX_MODEL_LEN = globals().get("MAX_MODEL_LEN", 65536)
MAX_GPU_HOURS = globals().get("MAX_GPU_HOURS", 12.0)
GPU_HOURLY_RATE_USD = globals().get("GPU_HOURLY_RATE_USD", None)
STARTUP_TIMEOUT_SECONDS = globals().get("STARTUP_TIMEOUT_SECONDS", 1800)
REPO_URL = globals().get(
    "REPO_URL", "https://github.com/gustavogomespl/agent-monitor-context-audit.git",
)
BRANCH = globals().get("BRANCH", "pilot")
CODE_REF = globals().get("CODE_REF", "")
PROJECT_ZIP = globals().get("PROJECT_ZIP", "")
SETUP_READY = globals().get("SETUP_READY", False)
apply_embedded_source = globals().get("apply_embedded_source", None)


def source_workspace():
    """Only explicit development amendments get separate source workspaces."""
    if EXPERIMENT_VERSION == "legacy":
        return DRIVE_ROOT
    if EXPERIMENT_VERSION in {"summary-v2", "summary-v3", "summary-v4", "summary-v5"}:
        return DRIVE_ROOT / "versions" / EXPERIMENT_VERSION
    raise ValueError("Unknown experiment version; choose the matching reviewed notebook.")


def prepare_version_workspace():
    """Inherit immutable setup choices once, without importing prior generation records."""
    import json
    import shutil

    workspace = source_workspace()
    if EXPERIMENT_VERSION == "legacy":
        return
    marker = workspace / "configuration/version.json"
    if marker.exists():
        if json.loads(marker.read_text()).get("experiment_version") != EXPERIMENT_VERSION:
            raise ValueError("Saved experiment version differs from this notebook.")
        return
    previous_versions = {
        "summary-v2": (),
        "summary-v3": ("summary-v2",),
        "summary-v4": ("summary-v2", "summary-v3"),
        "summary-v5": ("summary-v2", "summary-v3", "summary-v4"),
    }[EXPERIMENT_VERSION]
    older_workspaces = [DRIVE_ROOT, *(
        DRIVE_ROOT / "versions" / version for version in previous_versions
    )]
    frozen = any((older / name).exists() for older in older_workspaces for name in (
        "frozen-source.zip", "public-manifests/protocol-v1.json",
    ))
    test_runs = list((DRIVE_ROOT / "runs-private").glob("qwen-test*/manifests/run.json"))
    for manifest in (DRIVE_ROOT / "runs-private").rglob("manifests/run.json"):
        if json.loads(manifest.read_text()).get("config", {}).get("split") == "test":
            test_runs.append(manifest)
    if frozen or test_runs:
        raise ValueError("Prior frozen/test evidence requires review as a separate exploratory "
                         "study; this notebook amendment is for development only.")
    parent, parent_version = DRIVE_ROOT, "legacy"
    for version in reversed(previous_versions):
        previous = DRIVE_ROOT / "versions" / version
        previous_marker = previous / "configuration/version.json"
        if (previous_marker.exists()
                and json.loads(previous_marker.read_text()).get("experiment_version")
                == version):
            parent, parent_version = previous, version
            break
    inherited = [parent / "configuration/code-pin.json",
                 parent / "configuration/context/selection.json"]
    inherited.extend(path for path in (parent / "public-manifests").glob("*")
                     if path.is_file())
    for source in inherited:
        if not source.exists():
            continue
        target = workspace / source.relative_to(parent)
        # A retry after interrupted preparation must never overwrite partial setup.
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                raise ValueError("Incomplete version setup differs from its parent; "
                                 "review required.")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "experiment_version": EXPERIMENT_VERSION,
        "reason": ("Development summary floor 1024 and bounded citation schema amendment; "
                   "fresh complete pilot"
                   if EXPERIMENT_VERSION == "summary-v5" else
                   "Development summary cap 2048 amendment; fresh complete pilot"
                   if EXPERIMENT_VERSION == "summary-v4" else
                   "Development structured citation schema amendment; fresh complete pilot"
                   if EXPERIMENT_VERSION == "summary-v3" else
                   "Development summary length and citation amendment; fresh complete pilot"),
        "parent": parent_version, "shared_data_model_and_gpu_budget": True,
    }, indent=2) + "\n")
    temporary.replace(marker)


def numeric_results_root():
    root = DRIVE_ROOT / "numeric-results"
    return root if EXPERIMENT_VERSION == "legacy" else root / EXPERIMENT_VERSION


def status_directory():
    root = DRIVE_ROOT / "runs-private/notebook-status"
    return root if EXPERIMENT_VERSION == "legacy" else root / EXPERIMENT_VERSION


def mount_workspace():
    from google.colab import drive

    drive.mount("/content/drive")
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)


def release_gpu():
    from google.colab import runtime

    print("Releasing the Colab runtime. Results remain on Drive.", flush=True)
    try:
        runtime.unassign()
    except Exception:
        print("Automatic release failed. Use Runtime > Disconnect and delete runtime now.",
              flush=True)
        raise


def check_gpu():
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "No NVIDIA GPU is attached to this runtime. Use Runtime > Change runtime type, "
            "select an H100 80GB or RTX PRO 6000 GPU, then choose Run all again."
        ) from exc
    rows = result.stdout.strip().splitlines()
    if len(rows) != 1 or int(rows[0].rsplit(",", 1)[1]) < 75000:
        raise RuntimeError(
            f"Detected: {'; '.join(rows) or 'no GPU'}. Select exactly one GPU with at least "
            "75,000 MiB (H100 80GB or RTX PRO 6000) via Runtime > Change runtime type."
        )
    print("GPU:", rows[0], flush=True)

def bind_git_metadata(repo, durable_git, expected_commit=None):
    """Keep restored source files and their durable Git HEAD consistent."""
    import shutil
    import subprocess

    if expected_commit is not None and durable_git.exists():
        durable_head = subprocess.run(
            ["git", f"--git-dir={durable_git}", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if durable_head != expected_commit:
            raise ValueError(
                "Durable Git HEAD differs from the source commit; restore its matching snapshot."
            )
    local_git = repo / ".git"
    if local_git.is_symlink():
        if local_git.resolve() != durable_git.resolve():
            raise ValueError("Local source uses a different persistent Git workspace.")
    else:
        if durable_git.exists():
            if local_git.exists():
                shutil.rmtree(local_git)
        elif local_git.is_dir():
            shutil.copytree(local_git, durable_git)
            shutil.rmtree(local_git)
        else:
            raise ValueError("Source Git metadata is missing.")
        local_git.symlink_to(durable_git, target_is_directory=True)


def checkout_source(repo, repo_url, branch, code_ref, pin_path):
    """Select a branch once; preserve the exact source and local work on reconnect."""
    import json
    import re
    import subprocess

    if not repo_url or not branch or (code_ref and not re.fullmatch(r"[0-9a-f]{40}", code_ref)):
        raise ValueError("Provide a repository, a branch and an optional exact 40-character SHA.")
    valid = subprocess.run(
        ["git", "check-ref-format", "--branch", branch], capture_output=True, text=True
    )
    if valid.returncode:
        raise ValueError("Invalid source branch name.")
    saved = json.loads(pin_path.read_text()) if pin_path.exists() else None
    if saved is not None:
        if (
            saved.get("repo_url") != repo_url
            or saved.get("branch") != branch
            or not re.fullmatch(r"[0-9a-f]{40}", saved.get("commit", ""))
            or (code_ref and code_ref != saved["commit"])
        ):
            raise ValueError("Source selection differs from the saved pin; use a new workspace.")
    if repo.exists():
        if saved is None or not (repo / ".git").exists():
            raise ValueError("Existing checkout lacks a source pin; preserve it first.")
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        if head != saved["commit"]:
            raise ValueError("Existing HEAD differs from the source pin; no checkout was changed.")
        # Inventory manifests may be modified locally. Never reset or pull over them.
        return saved

    subprocess.run(["git", "clone", "--no-checkout", "--", repo_url, str(repo)], check=True)
    target = saved["commit"] if saved else code_ref or f"refs/heads/{branch}"
    subprocess.run(["git", "fetch", "origin", target], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "FETCH_HEAD^{commit}"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or (saved and commit != saved["commit"]):
        raise ValueError("Fetched source does not match its immutable commit.")
    subprocess.run(["git", "checkout", "--detach", commit], cwd=repo, check=True)
    if saved is None:
        saved = {"repo_url": repo_url, "branch": branch, "commit": commit}
        pin_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = pin_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(saved, indent=2) + "\n")
        temporary.replace(pin_path)
    return saved


def require_setup():
    if not SETUP_READY:
        raise RuntimeError("Environment preparation has not completed; run the workflow again.")


def safe_extract(archive_path, target):
    import stat
    import zipfile

    target = target.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            destination = (target / item.filename).resolve()
            if not destination.is_relative_to(target) or Path(item.filename).is_absolute():
                raise ValueError("Unsafe archive path; extraction refused.")
            if ".git" in Path(item.filename).parts or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Archive must contain ordinary source files, not Git metadata.")
        archive.extractall(target)


def bind_private_directory(relative, durable):
    import shutil

    link = REPO / relative
    durable.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != durable.resolve():
            raise ValueError("Existing private link points to another workspace.")
        return
    if link.exists():
        if any(link.iterdir()):
            raise ValueError("Existing local private data requires an explicit migration first.")
        shutil.rmtree(link)
    link.symlink_to(durable, target_is_directory=True)


def save_public_manifests():
    import shutil

    destination = source_workspace() / "public-manifests"
    destination.mkdir(parents=True, exist_ok=True)
    for source in (REPO / "data/manifests").glob("*"):
        if source.is_file():
            shutil.copy2(source, destination / source.name)


def verify_runtime_versions(pin_path, installed):
    import json

    pin = json.loads(pin_path.read_text())
    expected = pin.get("runtime_versions")
    if expected is not None and expected != installed:
        raise RuntimeError(
            "Inference dependencies differ from the saved runtime pin. Reinstall the exact "
            "pinned versions, or use a new explicitly exploratory workspace."
        )
    if expected is None:
        pin["runtime_versions"] = dict(installed)
        pin_path.write_text(json.dumps(pin, indent=2) + "\n")
    return pin


def prepare_vllm_imports():
    """Match the verified CUDA wheel variant, then import in a fresh process."""
    import importlib.metadata
    import json
    import subprocess
    import sys

    def versions():
        result = {}
        for name in ("torch", "vllm", "torchaudio"):
            try:
                result[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                result[name] = None
        return result

    before = versions()
    torch_probe = subprocess.run(
        [sys.executable, "-c", "import json, torch; "
         "print(json.dumps({'version': torch.__version__, 'cuda': torch.version.cuda}))"],
        capture_output=True, text=True, check=True, timeout=60,
    )
    torch_runtime = json.loads(torch_probe.stdout)
    # vLLM 0.28.0 pins TorchAudio 2.11.0 (stable ABI with Torch >=2.11).
    # A package version match alone can retain Colab's incompatible cu128 wheel.
    if (
        before["vllm"] == "0.28.0"
        and (before["torch"] or "").split("+")[0] == "2.13.0"
        and torch_runtime["cuda"] == "13.0"
        and before["torchaudio"] != "2.11.0+cu130"
    ):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps",
             "--only-binary=:all:", "--index-url", "https://download.pytorch.org/whl/cu130",
             "torchaudio==2.11.0+cu130"],
            check=True,
        )
    after = versions()
    if any(after[name] != before[name] for name in ("torch", "vllm")):
        raise RuntimeError("Auxiliary wheel repair changed the pinned Torch/vLLM versions.")
    probe = subprocess.run(
        [sys.executable, "-c", "import torchaudio; "
         "from vllm.entrypoints.openai import api_server; print('VLLM_IMPORT_OK')"],
        capture_output=True, text=True, timeout=120,
    )
    if probe.returncode:
        raise RuntimeError(
            "vLLM dependency import failed before model startup:\n"
            + (probe.stderr or probe.stdout)[-12000:]
        )
    print("VLLM_IMPORT_OK — dependency imports passed; no model started.")
    return dict(status="passed", before=before, after=after, torch_runtime=torch_runtime)


def prepare_structured_outputs():
    """Test decoder constraints on CPU before downloading or launching model weights."""
    import json
    import subprocess
    import sys

    if EXPERIMENT_VERSION not in {"summary-v3", "summary-v4", "summary-v5"}:
        return {"status": "not_requested"}
    probe = subprocess.run(
        [sys.executable, "-m", "context_audit.structured_backend"], cwd=REPO,
        capture_output=True, text=True, timeout=120,
    )
    if probe.returncode:
        raise RuntimeError("Citation decoder check failed before model startup:\n"
                           + (probe.stderr or probe.stdout)[-12000:])
    receipt = json.loads(probe.stdout.strip().splitlines()[-1])
    if receipt.get("status") != "passed" or receipt.get("model_generation_executed") is not False:
        raise RuntimeError("Citation decoder check returned an invalid receipt")
    print("STRUCTURED_OUTPUTS_OK — citation schema verified (xgrammar "
          + receipt["version"] + "); no model started.", flush=True)
    return receipt


def phase_settings(phase):
    """One durable context choice and distinct run identities across all stages."""
    import json

    if phase not in {"pilot", "development", "test"}:
        raise ValueError("Choose pilot, development, or test.")
    workspace = source_workspace()
    name = phase if EXPERIMENT_VERSION == "legacy" else f"{phase}-{EXPERIMENT_VERSION}"
    selection = workspace / "configuration/context/selection.json"
    if not selection.exists():
        return name, MAX_MODEL_LEN
    window = json.loads(selection.read_text())["context_window"]
    if type(window) is not int or not 2048 <= window <= 262144:
        raise ValueError("Invalid saved context selection; review the private configuration.")
    return f"{name}-ctx{window}", window


def phase_run_dir(phase):
    return Path("runs/private") / ("qwen-" + phase_settings(phase)[0])


def prepare_context_window():
    """Recover a complete token-only pilot preflight, without rewriting its run."""
    import hashlib
    import json

    from context_audit.cli import _dataset_manifest
    from context_audit.provider import utc_now
    from context_audit.runner import protocol_signature
    from context_audit.storage import PrivateStore

    if STAGE == "test":
        return False
    directory = DRIVE_ROOT / "runs-private" / phase_run_dir("pilot").name
    preflight_path = directory / "manifests/preflight.json"
    if not preflight_path.exists():
        return False
    preflight = json.loads(preflight_path.read_text())
    if not preflight.get("context_limit_ids"):
        return False
    if any(path.exists() for path in (
        source_workspace() / "frozen-source.zip",
        source_workspace() / "public-manifests/protocol-v1.json",
        REPO / "data/manifests/protocol-v1.json",
    )):
        raise ValueError("Context recovery cannot change a frozen protocol; review required.")
    # Include unsuccessful, pending and uncertain requests, not just successful scores.
    for run in (DRIVE_ROOT / "runs-private").glob("qwen-*"):
        if EXPERIMENT_VERSION != "legacy" and not any(
            run.name == f"qwen-{phase}-{EXPERIMENT_VERSION}"
            or run.name.startswith(f"qwen-{phase}-{EXPERIMENT_VERSION}-ctx")
            for phase in ("pilot", "development", "test")
        ):
            continue
        generation = any((run / name).exists() for name in (
            "scores.csv", "manifests/completion.json",
        )) or any(any((run / name).rglob("*")) for name in (
            "requests", "calls", "results", "representations",
        )) or any(path.exists() and path.read_text().strip() for path in (
            run / "budget-seconds.jsonl", run / "budget.jsonl",
        ))
        if generation:
            raise ValueError("Context recovery found generation evidence; preserve runs "
                             "for review.")
    name, _ = phase_settings("pilot")
    if not (source_workspace() / "configuration" / f"{name}.json").exists():
        raise ValueError("Context preflight lacks its saved configuration; review required.")
    config = configured_phase("pilot")
    manifest = json.loads((directory / "manifests/run.json").read_text())
    dataset = _dataset_manifest(config)
    signature = protocol_signature(config, dataset)
    for key, expected in (("config", config.model_dump()), ("code_hash", signature["code_hash"]),
                          ("prompt_hashes", signature["prompts"]),
                          ("dataset_manifest_hash", signature["dataset_manifest_hash"])):
        if manifest.get(key) != expected:
            raise ValueError("Context preflight methods or dataset differ; review required.")
    items = preflight["items"]
    if not items or len(items) != dataset["counts"]["eligible_transcripts"]:
        raise ValueError("Context inventory is incomplete; no window was inferred.")
    ids, problems, required = set(), set(), 0
    for item in items:
        identifier = item["transcript_id"]
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("Invalid or duplicate context inventory IDs.")
        ids.add(identifier)
        for key in ("body_tokens", "full_input_tokens", "summary_input_tokens"):
            if type(item[key]) is not int or item[key] < 0:
                raise ValueError("Invalid context token count; no window was inferred.")
        if (item["monitor_window"] != config.monitor_context_window
                or item["summary_window"] != config.summarizer_context_window):
            raise ValueError("Context inventory window differs from its saved configuration.")
        monitor = item["full_input_tokens"] + config.monitor_max_tokens
        summary = item["summary_input_tokens"] + config.summary_max_tokens
        required = max(required, monitor, summary)
        if monitor > item["monitor_window"] or summary > item["summary_window"]:
            problems.add(identifier)
    if (not problems or len(preflight["context_limit_ids"]) != len(problems)
            or set(preflight["context_limit_ids"]) != problems):
        raise ValueError("Context inventory limit IDs disagree with its token counts.")
    if required > 262144:
        raise ValueError(f"Full requests require {required} tokens, above the native 262144 "
                         "limit. Review scope/model; no transcripts were truncated or excluded.")
    # Native context only: 32K increments with up to 1K headroom for repair prefixes.
    selected = min(262144, ((required + 1024 + 32767) // 32768) * 32768)
    previous = config.qwen.max_model_len
    if selected <= previous:
        raise ValueError("Context recovery did not produce a larger window; review required.")
    record = dict(
        context_window=selected, previous_context_window=previous, required_tokens=required,
        source_run=str(directory.relative_to(DRIVE_ROOT)), source_run_id=manifest["run_id"],
        preflight_sha256=hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
        code_hash=signature["code_hash"], dataset_manifest_hash=signature["dataset_manifest_hash"],
        reason="Complete token inventory before any generation; retain all full inputs",
        selected_at=utc_now(),
    )
    store = PrivateStore(source_workspace() / "configuration")
    store.put("context", f"from-{previous}-to-{selected}", record)
    store.put("context", "selection", record)
    print(f"Context inventory: {len(items)} transcripts; maximum request plus output: "
          f"{required} tokens. Context: {previous} -> {selected}. "
          "Previous attempt retained; all full inputs preserved.", flush=True)
    return True


def configured_phase(phase):
    import json

    from context_audit.runtime_models import AuditConfig, QwenConfig

    require_setup()
    name, window = phase_settings(phase)
    if not DATA_USE_CONFIRMED or not RUBRIC_REVIEWED:
        raise ValueError("Confirm data-use compatibility and review the rubric before generation.")
    pin = json.loads((DRIVE_ROOT / "configuration/model-pin.json").read_text())
    backend = QwenConfig(
        model_revision=pin["model_revision"],
        vllm_version=pin["vllm_version"],
        runtime_versions=pin["runtime_versions"],
        base_url="http://127.0.0.1:8000",
        max_model_len=window,
        gpu_hourly_rate_usd=GPU_HOURLY_RATE_USD,
        gpu_budget_hours=MAX_GPU_HOURS,
    )
    backend.validate_live()
    config = AuditConfig(
        provider="qwen_local",
        qwen=backend,
        protocol_version=("protocol-v1" if phase == "test" else
                          "development-v1" if EXPERIMENT_VERSION == "legacy" else
                          f"development-{EXPERIMENT_VERSION}"),
        structured_summary_mode=(
            "schema_citations_bounded_v1" if EXPERIMENT_VERSION == "summary-v5" else
            "schema_citations_v1" if EXPERIMENT_VERSION in {"summary-v3", "summary-v4"}
            else "prompt"
        ),
        token_minimum=1024 if EXPERIMENT_VERSION == "summary-v5" else 128,
        token_maximum=2048 if EXPERIMENT_VERSION in {"summary-v4", "summary-v5"} else 1024,
        summary_max_tokens=3200 if EXPERIMENT_VERSION in {"summary-v4", "summary-v5"} else 1600,
        split="test" if phase == "test" else "development",
        dataset_dir="data/private",
        run_dir=str(phase_run_dir(phase)),
        monitor_model=pin["model_id"],
        summarizer_model=pin["model_id"],
        monitor_context_window=window,
        summarizer_context_window=window,
        pilot_pairs=3 if phase == "pilot" else None,
        timeout_seconds=300,
        data_use_confirmed=DATA_USE_CONFIRMED,
        rubric_reviewed=RUBRIC_REVIEWED,
        protocol_file="data/manifests/protocol-v1.json",
    )
    path = source_workspace() / "configuration" / f"{name}.json"
    payload = config.model_dump()
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError(
            "This phase already has a different saved configuration. Preserve its results and "
            "use a new explicit workspace/version for changed methods."
        )
    if not path.exists():
        path.write_text(json.dumps(payload, indent=2) + "\n")
    return config

def prepare_source():
    import json
    import shutil
    import subprocess
    import tempfile

    prepare_version_workspace()
    workspace = source_workspace()
    configuration = workspace / "configuration"
    configuration.mkdir(parents=True, exist_ok=True)
    saved_upload = workspace / "source-upload.zip"
    frozen_source = workspace / "frozen-source.zip"
    durable_git = workspace / "git-metadata"
    code_pin_path = configuration / "code-pin.json"
    source_kind = "git" if REPO_URL and not PROJECT_ZIP and not saved_upload.exists() else "bundle"
    if frozen_source.exists():
        source_kind = "frozen"
        if not durable_git.exists():
            raise ValueError("Frozen source lacks durable Git provenance; restore it first.")
        if not REPO.exists():
            REPO.mkdir(parents=True)
            safe_extract(frozen_source, REPO)
    elif source_kind == "git":
        checkout_source(REPO, REPO_URL, BRANCH, CODE_REF, code_pin_path)
    elif not REPO.exists():
        if not saved_upload.exists():
            if PROJECT_ZIP:
                source_zip = Path(PROJECT_ZIP)
            else:
                from google.colab import files

                uploaded = files.upload()
                if len(uploaded) != 1:
                    raise ValueError("Upload exactly one qwen-colab-bundle.zip.")
                source_zip = Path(next(iter(uploaded)))
            shutil.copy2(source_zip, saved_upload)
        with tempfile.TemporaryDirectory(prefix="qwen-source-") as staging:
            stage = Path(staging)
            safe_extract(saved_upload, stage)
            bundle = stage / "source.bundle"
            if not bundle.is_file() or not (stage / "research_plan.md").is_file():
                raise ValueError("Expected source.bundle and project files at the ZIP root.")
            subprocess.run(["git", "clone", str(bundle), str(REPO)], check=True)
            for source in stage.iterdir():
                if source.name == "source.bundle":
                    continue
                destination = REPO / source.name
                if source.is_dir():
                    shutil.copytree(source, destination, dirs_exist_ok=True)
                else:
                    shutil.copy2(source, destination)
    if not (REPO / "src/context_audit/colab.py").is_file():
        raise ValueError("This source revision does not include the Qwen Colab implementation.")
    expected_commit = (
        json.loads(code_pin_path.read_text())["commit"] if source_kind == "git" else None
    )
    bind_git_metadata(REPO, durable_git, expected_commit)
    bind_private_directory("data/private", DRIVE_ROOT / "data-private")
    bind_private_directory("runs/private", DRIVE_ROOT / "runs-private")
    saved_manifests = workspace / "public-manifests"
    if saved_manifests.exists():
        shutil.copytree(saved_manifests, REPO / "data/manifests", dirs_exist_ok=True)


    options = {} if EXPERIMENT_VERSION == "legacy" else {
        "run_root": DRIVE_ROOT / "runs-private", "run_version": EXPERIMENT_VERSION,
    }
    apply_embedded_source(REPO, workspace, frozen=source_kind == "frozen", **options)


def install_commands(pin):
    """Quiet pip commands: the pinned engine stack, then this project with its data extra."""
    import sys

    dependencies = [f"vllm=={pin['vllm_version']}", "transformers>=5.8.0,<6"]
    if EXPERIMENT_VERSION in {"summary-v3", "summary-v4", "summary-v5"}:
        dependencies.append("xgrammar==0.2.3")
    if "runtime_versions" in pin:
        dependencies.extend(
            f"{name}=={version}" for name, version in pin["runtime_versions"].items()
        )
    return [
        [sys.executable, "-m", "pip", "install", "-q", *dependencies],
        [sys.executable, "-m", "pip", "install", "-q", "-e", ".[data]"],
    ]


def install_runtime():
    global SETUP_READY
    SETUP_READY = False
    import importlib.metadata
    import json
    import re
    import subprocess
    import sys
    import urllib.parse
    import urllib.request
    from datetime import datetime, timezone

    configuration = DRIVE_ROOT / "configuration"
    configuration.mkdir(parents=True, exist_ok=True)
    code_pin_path = source_workspace() / "configuration/code-pin.json"
    pin_path = configuration / "model-pin.json"
    if pin_path.exists():
        pin = json.loads(pin_path.read_text())
        if pin["model_id"] != MODEL_ID or pin["vllm_version"] != VLLM_VERSION:
            raise ValueError("Model/engine differs from the durable pin; do not overwrite it.")
        if MODEL_REVISION and MODEL_REVISION != pin["model_revision"]:
            raise ValueError("Explicit model revision disagrees with the saved pin.")
    else:
        revision = MODEL_REVISION
        if not revision:
            model_path = urllib.parse.quote(MODEL_ID, safe="/")
            request = urllib.request.Request(
                f"https://huggingface.co/api/models/{model_path}/revision/main",
                headers={"User-Agent": "context-audit-colab-setup"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                revision = json.load(response)["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Model revision must be an exact immutable 40-character commit.")
        pin = {
            "model_id": MODEL_ID,
            "model_revision": revision,
            "vllm_version": VLLM_VERSION,
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        }
        pin_path.write_text(json.dumps(pin, indent=2) + "\n")
    engine, project = install_commands(pin)
    print("Installing the pinned inference packages quietly; this usually takes several "
          "minutes and only errors are printed.", flush=True)
    subprocess.run(engine, check=True)
    subprocess.run(project, cwd=REPO, check=True)
    inference_packages = ("vllm", "torch", "transformers", "tokenizers", "triton", "safetensors")
    installed_versions = {name: importlib.metadata.version(name) for name in inference_packages}
    pin = verify_runtime_versions(pin_path, installed_versions)
    dependency_import_check = prepare_vllm_imports()
    structured_output_check = prepare_structured_outputs()
    # CPU-visible provenance only; setup neither starts an engine nor queries a GPU.
    packages = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout
    setup_history = source_workspace() / "configuration/setup-history"
    setup_history.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    setup_record = {
        "schema_version": 1,
        "experiment_version": EXPERIMENT_VERSION,
        "created_at": timestamp,
        "python": sys.version,
        "code_commit": code_commit,
        "source_kind": "notebook_snapshot",
        "source_pin": json.loads(code_pin_path.read_text()) if code_pin_path.exists() else None,
        "model_pin": pin,
        "installed_packages": packages.splitlines(),
        "dependency_import_check": dependency_import_check,
        "structured_output_check": structured_output_check,
        "notebook_bootstrap_sha256": globals().get("NOTEBOOK_BOOTSTRAP_SHA256"),
    }
    (setup_history / f"{timestamp}.json").write_text(json.dumps(setup_record, indent=2) + "\n")
    sys.path.insert(0, str(REPO / "src"))
    # Pip and source replacement do not refresh modules imported in this kernel.
    for name in list(sys.modules):
        if name == "context_audit" or name.startswith("context_audit."):
            del sys.modules[name]
    import importlib
    importlib.invalidate_caches()
    SETUP_READY = True
    print("Source commit:", code_commit)
    print("python:", sys.version)
    print("Setup complete. Durable model revision:", pin["model_revision"])


def acquire_data():
    require_setup()
    from contextlib import chdir

    from context_audit.cli import main
    from context_audit.dataset import load_dataset

    with chdir(REPO):
        if not Path("data/private/manifest.json").exists():
            if main(["acquire"]):
                raise RuntimeError("Official dataset acquisition failed.")
            if main(["inventory"]):
                raise RuntimeError("Dataset inventory failed.")
        inputs, labels = load_dataset(Path("data/private"), "development")
        print("Validated development transcripts:", len(inputs))
        print("Existing opaque IDs and family split retained.")
        save_public_manifests()


def freeze_reviewed():
    import json
    import subprocess
    from contextlib import chdir

    from context_audit.cli import freeze

    with chdir(REPO):
        test_config = configured_phase("test")
        result = freeze(test_config, Path(configured_phase("development").run_dir))
        print(json.dumps(result, indent=2))
        tagged = subprocess.run(
            ["git", "rev-parse", "--verify", "protocol-v1^{commit}"],
            capture_output=True, text=True,
        )
        if tagged.returncode:
            public_paths = [
                "src", "tests", "prompts", "configs", "docs", "notebooks", "data/manifests",
                "scripts", "research_plan.md", "pyproject.toml", "uv.lock",
                ".gitignore", "README.md", "AGENTS.md",
            ]
            subprocess.run(["git", "add", "--all", "--", *public_paths], check=True)
            subprocess.run(
                [
                    "git", "-c", "user.name=Colab protocol snapshot",
                    "-c", "user.email=local-colab@invalid", "commit",
                    "-m", "Freeze reviewed Qwen protocol-v1",
                ], check=True,
            )
            subprocess.run(["git", "tag", "protocol-v1"], check=True)
        else:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip()
            if tagged.stdout.strip() != head:
                raise ValueError("Existing protocol-v1 does not identify HEAD; no tag was changed.")
        archive = source_workspace() / "frozen-source.zip"
        subprocess.run(
            ["git", "archive", "--format=zip", f"--output={archive}", "HEAD"], check=True
        )
        save_public_manifests()
        print("Reviewed protocol frozen locally. Starting the selected test stage next.")

def read_budget(root):
    """Read-only startup bound before installing any third-party dependencies."""
    import json
    import math

    budget = root / "runs-private/gpu_budget"
    pin_path = budget / "manifests/limit.json"
    pin = json.loads(pin_path.read_text()) if pin_path.exists() else {
        "limit_seconds": 43200.0, "previously_used_seconds": PRIOR_GPU_MINUTES * 60,
    }
    if pin["limit_seconds"] != 43200 or not 0 <= pin["previously_used_seconds"] <= 43200:
        raise ValueError("This notebook requires the saved cumulative 12-hour budget.")
    amounts, settled = {}, set()
    journal = budget / "budget-seconds.jsonl"
    if journal.exists():
        for line in journal.read_text().splitlines():
            entry = json.loads(line)
            key, amount = entry["call_id"], entry["amount"]
            if not math.isfinite(amount) or amount < 0:
                raise ValueError("Invalid saved GPU accounting; review the private ledger.")
            if entry["action"] == "reserve" and key not in amounts:
                amounts[key] = amount
            elif entry["action"] == "settle" and key in amounts:
                settled.add(key)
                amounts[key] = amount
            else:
                raise ValueError("Invalid saved GPU accounting; review the private ledger.")
    # Include the initial debit even if an interruption preceded its first journal entry.
    used = sum(amounts.values())
    if "previously_used_gpu_time" not in amounts:
        used += pin["previously_used_seconds"]
    return dict(pin=pin, committed_seconds=used,
                remaining_seconds=max(0, 43200 - used),
                uncertain_sessions=sorted(set(amounts) - settled))


class NotebookAllocation:
    """Bound this workflow's allocation and debit measured non-supervisor time.

    A stopped Python kernel leaves a durable unresolved workflow receipt. A known
    setup failure keeps its measured overhead for the next successful installation.
    The core supervisor retains its own independent process watchdog and ledger.
    """

    def __init__(self, root, started, disconnect):
        import json
        import threading
        import time
        import uuid

        self.root = root
        self.run_dir = root / "runs-private/qwen-pilot"
        self.started = started or time.monotonic()
        self.mark = self.started
        self.count = 0
        self.pending = []
        snapshot = read_budget(root)
        if snapshot["uncertain_sessions"]:
            raise ValueError("An interrupted GPU session needs verified time reconciliation.")
        self.prior = snapshot["pin"]["previously_used_seconds"]
        self.committed = snapshot["committed_seconds"]
        history = root / "runs-private/notebook-sessions"
        history.mkdir(parents=True, exist_ok=True)
        for path in history.glob("*.json"):
            saved = json.loads(path.read_text())
            if saved["status"] == "running":
                raise ValueError("Previous notebook end time is unknown; reconcile its private "
                                 "notebook-sessions receipt before resuming.")
            if saved["status"] == "stopped_unaccounted":
                self.pending.append((path, saved))
        remaining = snapshot["remaining_seconds"] - sum(
            s["unaccounted_seconds"] for _, s in self.pending
        ) - (time.monotonic() - self.started)
        if remaining <= 30:
            raise ValueError("The cumulative GPU budget is exhausted; no new run was started.")
        self.deadline = time.monotonic() + remaining - 10
        self.id = uuid.uuid4().hex
        self.path = history / f"{self.id}.json"
        self.record = dict(status="running",
                           started_epoch=time.time() - (time.monotonic() - self.started),
                           deadline_epoch=time.time() + remaining,
                           committed_before=self.committed)
        self.write()
        self.timer = threading.Timer(max(1, remaining - 5), disconnect)
        self.timer.daemon = True
        self.timer.start()

    def write(self):
        import json

        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.record, indent=2) + "\n")
        temporary.replace(self.path)

    def remaining(self):
        import time

        return max(0, self.deadline - time.monotonic())

    def checkpoint(self):
        import time

        from context_audit.colab import initialize_gpu_budget, record_external_gpu_time

        receipt = initialize_gpu_budget(self.run_dir, 12, previously_used_seconds=self.prior)
        for path, saved in self.pending:
            previous_committed = receipt["committed_seconds"]
            receipt = record_external_gpu_time(
                self.run_dir, 12, usage_id=f"notebook-{path.stem}-recovery",
                elapsed_seconds=saved["unaccounted_seconds"], confirmed=True,
            )
            saved["status"] = "accounted"
            import json
            path.write_text(json.dumps(saved, indent=2) + "\n")
            self.committed += receipt["committed_seconds"] - previous_committed
        self.pending.clear()
        receipt = initialize_gpu_budget(self.run_dir, 12, previously_used_seconds=self.prior)
        now = time.monotonic()
        # Core sessions have already charged their startup, inference and teardown.
        overhead = max(0, now - self.mark - (receipt["committed_seconds"] - self.committed))
        receipt = record_external_gpu_time(
            self.run_dir, 12, usage_id=f"notebook-{self.id}-{self.count}",
            elapsed_seconds=overhead, confirmed=True,
        )
        self.mark, self.committed = now, receipt["committed_seconds"]
        self.count += 1
        self.record.update(accounted_through_seconds=now - self.started,
                           committed_after=self.committed)
        self.write()
        return receipt

    def close(self):
        import time

        try:
            self.checkpoint()
            self.record["status"] = "accounted"
        except Exception:
            # Dependency installation can fail before the ledger API is importable.
            snapshot = read_budget(self.root)
            self.record.update(
                status="stopped_unaccounted",
                unaccounted_seconds=max(0, time.monotonic() - self.mark
                                        - (snapshot["committed_seconds"] - self.committed)),
            )
            print("Measured setup time saved for accounting on the next successful setup.")
        finally:
            self.record["finished_epoch"] = time.time()
            self.write()
            # The caller releases the runtime immediately after this method.
            self.timer.cancel()


def phase_complete(config):
    """Reuse only an intact successful run with the same scientific signature."""
    import hashlib
    import json

    from context_audit.cli import _dataset_manifest
    from context_audit.runner import protocol_signature

    directory = Path(config.run_dir)
    completion = directory / "manifests/completion.json"
    manifest = directory / "manifests/run.json"
    scores = directory / "scores.csv"
    if not all(p.exists() for p in (completion, manifest, scores)):
        return False
    saved = json.loads(manifest.read_text())
    signature = protocol_signature(config, _dataset_manifest(config))
    for key, expected in (("code_hash", signature["code_hash"]),
                          ("dataset_manifest_hash", signature["dataset_manifest_hash"]),
                          ("prompt_hashes", signature["prompts"]), ("config", config.model_dump())):
        if saved[key] != expected:
            raise ValueError("Saved run methods differ. Preserve this workspace for review.")
    state = json.loads(completion.read_text())
    if state.get("status") == "executed" and state.get("scores_sha256") != hashlib.sha256(
        scores.read_bytes()
    ).hexdigest():
        raise ValueError("Completed scores changed; preserve the files for integrity review.")
    return (
        state.get("status") == "executed" and state.get("run_id") == saved["run_id"]
        and state.get("rows") == state.get("successful_rows") == state.get("expected_rows")
        == len(saved["planned_calls"])
        and state.get("scores_sha256") == hashlib.sha256(scores.read_bytes()).hexdigest()
    )


def execute_phase(config, seconds):
    import csv
    import json
    import threading
    import time

    from context_audit.colab import run_colab_experiment

    stopped = threading.Event()
    started = time.monotonic()

    def progress():
        while not stopped.wait(30):
            directory = Path(config.run_dir)
            elapsed = (time.monotonic() - started) / 60
            try:
                manifest = directory / "manifests/run.json"
                scores = directory / "scores.csv"
                if manifest.exists() and scores.exists():
                    total = len(json.loads(manifest.read_text())["planned_calls"])
                    with scores.open() as handle:
                        rows = list(csv.DictReader(handle))
                    good = sum(row["status"] == "ok" for row in rows)
                    print(f"  {elapsed:.1f} min | successful evaluations: {good}/{total}",
                          flush=True)
                else:
                    print(f"  {elapsed:.1f} min | loading model / checking context lengths...",
                          flush=True)
            except (OSError, ValueError, KeyError):
                pass  # A concurrent atomic score update will be read next time.

    observer = threading.Thread(target=progress, daemon=True)
    observer.start()
    try:
        return run_colab_experiment(
            config, max_cost_usd=None, session_max_seconds=seconds,
            startup_timeout_seconds=min(STARTUP_TIMEOUT_SECONDS, seconds - 15),
        )
    finally:
        stopped.set()
        observer.join(timeout=2)


def export_phase(phase, seconds):
    """Bound postprocessing in a child; private logs never become public outputs."""
    import json
    import subprocess
    import sys
    import time

    if seconds <= 1:
        raise ValueError("No remaining time for reports. Reproduce the saved scores on CPU.")
    run_dir = phase_run_dir(phase)
    numeric = numeric_results_root() / phase
    numeric.mkdir(parents=True, exist_ok=True)
    commands = []
    if (run_dir / "scores.csv").exists():
        commands.append(["export", "--run-dir", str(run_dir), "--output", str(numeric)])
    else:
        return {"status": "no_scores", "run_dir": str(run_dir)}
    commands.append(["analyze", "--scores", str(numeric / "public_scores.csv"),
                     "--manifest", str(numeric / "run_manifest.json"),
                     "--output", str(numeric / "reproduced")])
    deadline = time.monotonic() + seconds
    with (numeric / "analysis.log").open("w") as log:
        for command in commands:
            subprocess.run(
                [sys.executable, "-m", "context_audit.cli", *command], cwd=REPO,
                check=True, stdout=log, stderr=log,
                timeout=max(1, deadline - time.monotonic()),
            )
    report = numeric / "reproduced/findings.md"
    print("Report:", report)
    print("Scores:", numeric / "public_scores.csv")
    metrics = json.loads((numeric / "reproduced/metrics.json").read_text())
    if metrics.get("status") == "analyzed":
        print("AUROC on this stage (see the report for coverage and confidence intervals):")
        for condition in ("full", "head_tail", "free_summary", "structured_summary"):
            estimate = metrics["conditions"][condition]["auroc"]["estimate"]
            print(f"  {condition}: {estimate:.3f}" if estimate is not None
                  else f"  {condition}: unavailable")
    return dict(report=str(report), scores=str(numeric / "public_scores.csv"))


def private_log_tail(phase):
    """Last 64 KB of the latest private engine and worker logs; never printed verbatim."""
    sessions = DRIVE_ROOT / "runs-private" / phase_run_dir(phase).name / "gpu_sessions"
    details = ""
    for kind in ("server_logs", "runner_logs"):
        logs = list((sessions / kind).glob("*.json"))
        if logs:
            latest = max(logs, key=lambda path: path.stat().st_mtime)
            with latest.open("rb") as handle:
                handle.seek(max(0, latest.stat().st_size - 64000))
                tail = handle.read().decode(errors="replace")
            details += f"\n--- Private {kind}: {latest.name} ---\n" + tail
    return details


def failure_hints(details):
    """Short content-free explanations recognized in a private diagnostic text."""
    hints = []
    for line in details.splitlines():
        # The worker CLI reports its own handled errors on one prefixed line.
        if line.startswith("context-audit: ") and len(hints) < 5:
            if "validation error" in line:
                hints.append("context-audit: a validation error occurred; its field details "
                             "stay in the private log.")
            else:
                hints.append(line[:300])
    if "compiled with different CUDA versions" in details:
        hints.append("Dependency diagnosis: Torch and TorchAudio CUDA builds still differ.")
    if ("FlashInfer requires GPUs with sm75 or higher" in details
            and "topk_topp_sampler" in details):
        hints.append("FlashInfer sampler failed its architecture check during startup. "
                     "Use the updated Qwen notebook, which selects the native PyTorch sampler.")
    if "out of memory" in details.lower():
        hints.append("GPU memory was insufficient. Review the pilot settings before retrying.")
    if "exceed context" in details:
        hints.append("Some full transcripts exceed the configured context window; no truncation "
                     "was applied. Review the development context setting.")
    if "end time is unknown" in details or "verified time reconciliation" in details:
        hints.append("Previous allocation time is uncertain. Reconcile its receipt before "
                     "resuming.")
    return hints


def describe_error(error):
    """One safe line: the class and first message line, never validation field details."""
    name = type(error).__name__
    if name == "ValidationError":
        return f"{name} (field details are kept in the private diagnostic)"
    lines = str(error).strip().splitlines()
    return f"{name}: {lines[0][:300]}" if lines and lines[0] else name


def describe_partial_phase(phase, config, summary):
    """Content-free progress summary of an incomplete phase and where its logs are."""
    import csv
    import json
    from collections import Counter

    run_dir = Path(config.run_dir)
    print(f"{phase} did not complete (runner exit code {summary.get('exit_code')}).",
          flush=True)
    completion = run_dir / "manifests/completion.json"
    if completion.exists():
        state = json.loads(completion.read_text())
        print(f"  successful evaluations: {state.get('successful_rows')}/"
              f"{state.get('expected_rows')}", flush=True)
    scores = run_dir / "scores.csv"
    if scores.exists():
        with scores.open() as handle:
            counts = Counter(row.get("status", "") for row in csv.DictReader(handle))
        print("  evaluation statuses: "
              + ", ".join(f"{status}: {count}" for status, count in sorted(counts.items())),
              flush=True)
    for hint in failure_hints(private_log_tail(phase)):
        print("  " + hint, flush=True)
    print("  Private engine/worker logs:",
          DRIVE_ROOT / "runs-private" / phase_run_dir(phase).name / "gpu_sessions", flush=True)


def form_checklist():
    """Which first-form confirmations are still missing for the selected stage."""
    fields = [("DATA_USE_CONFIRMED", DATA_USE_CONFIRMED), ("RUBRIC_REVIEWED", RUBRIC_REVIEWED)]
    if STAGE == "test":
        fields.append(("DEVELOPMENT_REVIEWED", DEVELOPMENT_REVIEWED))
    fields.append(("START_RUN", START_RUN))
    return "\n".join(f"  [{'x' if value else ' '}] {name}" for name, value in fields)


def record_status(result, error=None):
    import json
    import traceback

    folder = status_directory()
    folder.mkdir(parents=True, exist_ok=True)
    if error is not None:
        phase = result.get("active_phase", STAGE)
        details = "".join(traceback.format_exception(error)) + private_log_tail(phase)
        (folder / "last-error.log").write_text(details)
        for hint in failure_hints(details):
            print(hint, flush=True)
    (folder / "latest.json").write_text(json.dumps(result, indent=2) + "\n")


def run_guided():
    """One explicit form submission runs one stage; no hidden test-set progression."""
    import time
    from contextlib import chdir

    global SETUP_READY

    if not START_RUN:
        print(f"Not started (STAGE = {STAGE}). Complete form 1, then choose Runtime > Run all:\n"
              + form_checklist())
        return {"status": "not_started"}
    if not DATA_USE_CONFIRMED or not RUBRIC_REVIEWED:
        raise ValueError("Please confirm data use and the rubric in the first form.\n"
                         + form_checklist())
    if STAGE not in {"pilot", "development", "test"}:
        raise ValueError("Select pilot, development or test in the form.")
    if STAGE == "test" and not DEVELOPMENT_REVIEWED:
        raise ValueError("Test requires review of the completed development results.")
    if not isinstance(PRIOR_GPU_MINUTES, (int, float)) or not 0 <= PRIOR_GPU_MINUTES <= 720:
        raise ValueError("Initial prior GPU minutes must be between 0 and 720.")

    SETUP_READY = False
    started, allocation = time.monotonic(), None
    result = dict(status="preparing", stage=STAGE, experiment_version=EXPERIMENT_VERSION, phases={})
    print(f"Experiment: {EXPERIMENT_VERSION} | Notebook build: "
          f"{globals().get('NOTEBOOK_BOOTSTRAP_SHA256', 'source')[:12]}", flush=True)
    print(f"Stage: {STAGE} | Workspace: {DRIVE_ROOT} | Model: {MODEL_ID} (vLLM {VLLM_VERSION})"
          f" | Shared budget: {MAX_GPU_HOURS:g} GPU hours", flush=True)
    print("Plan: GPU check > Drive + budget > source + dependencies > dataset > inference > "
          "report > disconnect. Google Drive access is the only expected prompt.", flush=True)
    try:
        print("[1/6] Checking the GPU runtime.", flush=True)
        check_gpu()
        print("[2/6] Connecting Drive and restoring the shared 12-hour budget.", flush=True)
        mount_workspace()
        allocation = NotebookAllocation(DRIVE_ROOT, started, release_gpu)
        print("[3/6] Preparing matching source and checking inference dependencies.", flush=True)
        prepare_source()
        install_runtime()
        budget = allocation.checkpoint()
        print(f"GPU budget remaining: {budget['remaining_seconds'] / 3600:.2f} hours.")
        print("[4/6] Acquiring or validating the official paired dataset.", flush=True)
        acquire_data()
        phases = ["pilot", "development"] if STAGE == "development" else [STAGE]
        with chdir(REPO):
            prepare_context_window()
            if STAGE == "test":
                freeze_reviewed()
            for phase in phases:
                result["active_phase"] = phase
                config = configured_phase(phase)
                budget = allocation.checkpoint()
                available = min(budget["remaining_seconds"], allocation.remaining())
                print(f"[5/6] {phase}: checking saved progress and running four conditions.",
                      flush=True)
                print("  Run:", config.run_dir, flush=True)
                if phase_complete(config):
                    summary = {"status": "executed", "reused_completed_run": True}
                    print(f"  {phase} is already complete; reusing its saved evaluations.",
                          flush=True)
                else:
                    # Leave bounded time for reports and notebook teardown.
                    if available <= 180:
                        raise ValueError("Insufficient GPU time for a run and its report.")
                    print("  Starting the local vLLM server with the native PyTorch sampler. "
                          "The first session downloads about "
                          "55 GB of weights before scoring; progress prints every 30 seconds.",
                          flush=True)
                    summary = execute_phase(config, available - 150)
                    if (phase == "pilot" and summary["status"] != "executed"
                            and prepare_context_window()):
                        # At most one restart, only after complete token-only preflight.
                        budget = allocation.checkpoint()
                        available = min(budget["remaining_seconds"], allocation.remaining())
                        if available <= 180:
                            raise ValueError("Context saved; insufficient GPU time to restart.")
                        config = configured_phase(phase)
                        print("  Restarting the pilot with the measured context window.",
                              flush=True)
                        summary = execute_phase(config, available - 150)
                result["phases"][phase] = summary
                allocation.checkpoint()
                if summary["status"] != "executed":
                    describe_partial_phase(phase, config, summary)
                print(f"[6/6] Saving {phase} scores, figures and report on Drive.", flush=True)
                summary["outputs"] = export_phase(phase, min(120, allocation.remaining()))
                result["status"] = summary["status"]
                if summary["status"] != "executed":
                    print("Run incomplete. Saved records require review before advancing.")
                    break
        record_status(result)
        print("Finished:", result["status"], "| Results:", numeric_results_root())
        if STAGE == "development" and result["status"] == "executed":
            print("Review the development report before selecting test in a later run.")
        return result
    except Exception as error:
        result["status"] = "failed"
        reason = describe_error(error)
        print("Stopped:", reason, flush=True)
        record_status(result, error)
        print("Full private diagnostic:",
              status_directory() / "last-error.log")
        phase = result.get("active_phase", STAGE)
        print("Server logs:",
              DRIVE_ROOT / "runs-private" / phase_run_dir(phase).name / "gpu_sessions")
        raise RuntimeError(
            f"Workflow stopped: {reason}. Inspect the saved private diagnostic above."
        ) from None
    finally:
        try:
            if allocation is not None:
                allocation.close()
        finally:
            release_gpu()
