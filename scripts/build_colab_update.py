"""Package changed public source plus its guarded standalone Colab updater."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

HARDWARE_PATCH_SHA256 = "b01cee9beca6b25763d800f4ba58911134f11777b84e1e7fced4359cffdbcced"


def _updater():
    path = Path(__file__).with_name("apply_colab_update.py")
    spec = importlib.util.spec_from_file_location("colab_source_updater", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True
    ).stdout


def build_update(repo: Path, output: Path, *, compatibility_patch: Path | None = None) -> dict:
    repo, output = repo.resolve(), output.resolve()
    updater = _updater()
    base = _git(repo, "rev-parse", "HEAD").decode().strip()
    names = set(_git(repo, "diff", "--name-only", "-z", base, "--").decode().split("\0"))
    names.update(
        _git(repo, "ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    )
    paths = sorted(name for name in names if updater.source_path_allowed(name))
    if not paths:
        raise ValueError("No changed public source files to package")
    variants = {}
    if compatibility_patch is not None:
        patch = compatibility_patch.resolve()
        if updater.sha256(patch.read_bytes()) != HARDWARE_PATCH_SHA256:
            raise ValueError("Compatibility patch differs from the previously supplied artifact")
        name = "src/context_audit/colab.py"
        original = _git(repo, "show", f"{base}:{name}")
        with tempfile.TemporaryDirectory(prefix="colab-source-compat-") as temporary:
            staging = Path(temporary)
            _git(staging, "init", "--quiet")
            target = staging / name
            target.parent.mkdir(parents=True)
            target.write_bytes(original)
            _git(staging, "apply", "--check", str(patch))
            _git(staging, "apply", str(patch))
            variants[name] = updater.sha256(target.read_bytes())
    files, payload = [], {}
    for name in paths:
        source = repo / name
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Deleted or nonordinary source files cannot be packaged: {name}")
        body = source.read_bytes()
        tracked = subprocess.run(
            ["git", "-C", str(repo), "show", f"{base}:{name}"], capture_output=True
        )
        accepted = [updater.sha256(tracked.stdout) if tracked.returncode == 0 else None]
        if name in variants:
            accepted.append(variants[name])
        files.append({"path": name, "sha256": updater.sha256(body), "accepted_sha256": accepted})
        payload[f"payload/{name}"] = body
    manifest = {"schema_version": 1, "base_commit": base, "files": files}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
        archive.writestr("apply_update.py", Path(updater.__file__).read_bytes())
        for name, body in payload.items():
            archive.writestr(name, body)
    digest = updater.sha256(output.read_bytes())
    cell = output.with_name("colab-update-cell.py")
    cell.write_text(update_cell(digest))
    return {
        "archive": str(output), "sha256": digest, "update_cell": str(cell),
        "base_commit": base, "file_count": len(files),
    }


def update_cell(archive_sha256: str) -> str:
    """Generate a pasteable, opt-in migration cell bound to this exact archive."""
    return f'''# Paste this complete cell into the existing Colab notebook.
# Upload qwen-time-budget-update.zip when prompted. This cell starts no model.
import hashlib
import importlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

from google.colab import files

if "REPO" not in globals() or not globals().get("SETUP_READY", False):
    raise RuntimeError("Use this update in the existing runtime after completed setup.")
_uploaded_update = files.upload()
if len(_uploaded_update) != 1:
    raise ValueError("Upload exactly one qwen-time-budget-update.zip.")
_update_bytes = next(iter(_uploaded_update.values()))
if hashlib.sha256(_update_bytes).hexdigest() != {archive_sha256!r}:
    raise ValueError("The uploaded update differs from the reviewed archive SHA256.")
with tempfile.TemporaryDirectory(prefix="context-audit-update-") as _update_staging:
    _update_archive = Path(_update_staging) / "qwen-time-budget-update.zip"
    _update_archive.write_bytes(_update_bytes)
    with zipfile.ZipFile(_update_archive) as _update_zip:
        _updater_source = _update_zip.read("apply_update.py").decode("utf-8")
    _updater_namespace = {{"__name__": "reviewed_colab_source_update"}}
    exec(compile(_updater_source, "reviewed-apply-update.py", "exec"), _updater_namespace)
    _update_receipt = _updater_namespace["apply_update"](Path(REPO), _update_archive)
print("Source update applied; pinned Git HEAD preserved:", _update_receipt["base_commit"])

# Refresh only project modules. Do not reload Torch/vLLM or rerun setup.
for _module_name in list(sys.modules):
    if _module_name == "context_audit" or _module_name.startswith("context_audit."):
        del sys.modules[_module_name]
importlib.invalidate_caches()
_updated_notebook = json.loads((Path(REPO) / "notebooks/03_qwen_colab.ipynb").read_text())
_updated_code_cells = [
    "".join(cell["source"]) for cell in _updated_notebook["cells"]
    if cell["cell_type"] == "code"
]
_updated_helpers = next(
    source for source in _updated_code_cells if "def configured_phase(" in source
)
_updated_live_source = next(
    source for source in _updated_code_cells if "# LIVE_EXECUTION" in source
)
exec(compile(_updated_helpers, "updated-colab-helpers", "exec"), globals())

MAX_GPU_HOURS = 12.0
GPU_ALREADY_USED_HOURS = globals().get("GPU_ALREADY_USED_HOURS", 0.0)
GPU_ADDITIONAL_USED_HOURS = globals().get("GPU_ADDITIONAL_USED_HOURS", 0.0)
GPU_ADDITIONAL_USAGE_ID = globals().get("GPU_ADDITIONAL_USAGE_ID", "")
MAX_COST_USD = None
GPU_HOURLY_RATE_USD = None
SESSION_MAX_SECONDS = globals().get("SESSION_MAX_SECONDS", 3600)
STARTUP_TIMEOUT_SECONDS = globals().get("STARTUP_TIMEOUT_SECONDS", 900)
RUN_LIVE = False

def run_updated_pilot():
    if globals().get("PHASE") != "pilot" or not globals().get("RUN_LIVE", False):
        raise ValueError('Set PHASE="pilot" and RUN_LIVE=True after completing your reviews.')
    exec(compile(_updated_live_source, "updated-colab-live", "exec"), globals())

print("SETUP_READY preserved:", SETUP_READY)
print("No model started. Enter prior GPU hours, confirm data use/rubric after review,")
print('then set PHASE="pilot", RUN_LIVE=True and call run_updated_pilot().')
print("Use the updated notebook in a CPU runtime for later exports and analysis.")
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compatibility-patch", type=Path)
    args = parser.parse_args()
    output = args.output or args.repo / "dist/qwen-time-budget-update.zip"
    patch = args.compatibility_patch
    default_patch = args.repo / "dist/colab-gpu-compat.patch"
    if patch is None and default_patch.is_file():
        patch = default_patch
    print(json.dumps(build_update(args.repo, output, compatibility_patch=patch), indent=2))


if __name__ == "__main__":
    main()
