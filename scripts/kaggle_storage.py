"""Place notebook runtimes outside saved job output and report filesystem space."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile


GIB = 1024 ** 3


@dataclass(frozen=True)
class NotebookStorage:
    root: Path
    output: Path
    mode: str
    project: Path

    @property
    def python_cache(self) -> Path:
        return self.root / "kaggle-python"

    @property
    def studio_env(self) -> Path:
        return self.root / "kaggle-studio" / "venv"

    @property
    def vllm_env(self) -> Path:
        return self.root / "kaggle-vllm" / "venv"

    def cache_environment(self) -> dict[str, str]:
        """Overrides for child processes. Keep explicitly configured model caches."""
        if self.mode == "checkout":
            return {}
        result = {"TMPDIR": str(self.root / "tmp")}
        user_cache = Path.home() / ".cache"
        reuse_defaults = not user_cache.resolve().is_relative_to(self.output)
        hf_configured = any(os.environ.get(name) for name in
                            ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"))
        if not hf_configured and not (reuse_defaults and (user_cache / "huggingface/hub").is_dir()):
            result["HF_HUB_CACHE"] = str(self.root / "models" / "huggingface")
        if not (os.environ.get("TORCH_HOME") or os.environ.get("XDG_CACHE_HOME")
                or (reuse_defaults and (user_cache / "torch").is_dir())):
            result["TORCH_HOME"] = str(self.root / "models" / "torch")
        return result


def _temporary_root(project: Path, output: Path) -> Path:
    # A stable checkout-specific path lets a rerun reuse the same downloads.
    kaggle_temp = Path("/kaggle/temp")
    base = Path(tempfile.gettempdir())
    if base.resolve().is_relative_to(output) or base.resolve().is_relative_to(project):
        base = Path("/tmp")
    if output == Path("/kaggle/working"):
        try:
            kaggle_temp.mkdir(exist_ok=True)
        except OSError:
            pass  # Some runtimes only provide the system temporary directory.
        else:
            if os.access(kaggle_temp, os.W_OK | os.X_OK):
                base = kaggle_temp
    checkout = hashlib.sha256(os.fsencode(project)).hexdigest()[:12]
    return base / f"opensource-clipping-{checkout}"


def prepare_storage(project: Path, workspace: Path, *, mode: str = "temporary",
                    runtime_root: Path | str | None = None) -> NotebookStorage:
    """Select paths before either the managed Python or app packages are installed."""
    project = Path(project).resolve(strict=True)
    output = Path(workspace).resolve(strict=True)
    if not project.is_dir() or not output.is_dir():
        raise ValueError("The checkout and output workspace must be directories.")
    legacy_root = project / ".cache"
    if mode == "checkout":
        if runtime_root is not None:
            raise ValueError("Set runtime_root only in temporary storage mode.")
        root = legacy_root
    elif mode == "temporary":
        root = Path(runtime_root).expanduser() if runtime_root is not None else _temporary_root(project, output)
        if not root.is_absolute():
            raise ValueError("runtime_root must be an absolute path to a writable runtime directory.")
        root = root.resolve()
        if any(root.is_relative_to(path) or path.is_relative_to(root) for path in (project, output)):
            raise ValueError("Choose a runtime directory outside the checkout and output workspace.")
        if root.is_relative_to(Path("/kaggle/input")):
            raise ValueError("Choose a writable runtime directory outside /kaggle/input.")
        previous = [legacy_root / name for name in
                    ("kaggle-python", "kaggle-studio/venv", "kaggle-studio-cpu/venv", "kaggle-vllm/venv")]
        for path in previous:
            if path.is_dir():
                print(f"Existing checkout runtime still uses storage: {path}")
    else:
        raise ValueError("Storage mode must be 'temporary' or 'checkout'.")
    try:
        root.mkdir(parents=True, exist_ok=True)
        if mode == "temporary":
            (root / "tmp").mkdir(exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Cannot prepare runtime directory {root}: {exc}") from exc
    return NotebookStorage(root=root, output=output, mode=mode, project=project)


def check_storage(layout: NotebookStorage, min_output_gib: float = 1.0) -> None:
    """Check reported free space at this point in time. Video sizes vary by job."""
    if not math.isfinite(min_output_gib) or min_output_gib < 0:
        raise ValueError("min_output_gib must be finite and non-negative.")
    runtime_free = shutil.disk_usage(layout.root).free
    output_free = shutil.disk_usage(layout.output).free
    print(f"Runtime: {layout.root} ({runtime_free / GIB:.2f} GiB free)")
    print(f"Output:  {layout.output} ({output_free / GIB:.2f} GiB free, "
          f"{min_output_gib:g} GiB requested for jobs)")
    if layout.root.stat().st_dev == layout.output.stat().st_dev:
        print("Runtime and output share a filesystem. Both draw on its free space.")
    destinations = [(layout.output, output_free)]
    job_directory = layout.project / "outputs"
    job_volume = job_directory if job_directory.is_dir() else layout.project
    if job_volume.stat().st_dev != layout.output.stat().st_dev:
        job_free = shutil.disk_usage(job_volume).free
        print(f"Job videos: {job_directory} ({job_free / GIB:.2f} GiB free)")
        destinations.append((job_directory, job_free))
    for destination, free in destinations:
        if free < min_output_gib * GIB:
            raise RuntimeError(
                f"Insufficient output space at {destination}: {free / GIB:.2f} GiB free, "
                f"{min_output_gib:g} GiB requested. Save needed results, then free space or use "
                "a fresh session before downloading video."
            )
