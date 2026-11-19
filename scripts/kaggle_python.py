"""Select a Python 3.12 bootstrap without modifying the Kaggle kernel.

Only ensure_python() may download/install anything. It reuses an available
CPython 3.12 or asks a checksum-pinned uv binary to install one in the chosen cache.
The application and vLLM helpers still create their own separate virtualenvs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / ".cache" / "kaggle-python"
UV_VERSION = "0.13.9"
# Published release-asset digest from astral-sh/uv 0.13.9.
UV_SHA256 = "a7fbaff2b9a271d85c1bdb540941090fd1ce011797b4abeb48e4dcd39c2471d5"
UV_MEMBER = "uv-x86_64-unknown-linux-gnu/uv"
UV_URL = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"


def bootstrap_environment(cache: Path) -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("UV_", "PYTHON", "PIP_")) and key != "VIRTUAL_ENV"
    }
    environment.update({
        "UV_PYTHON_INSTALL_DIR": str(cache / "python"),
        "UV_CACHE_DIR": str(cache / "uv-cache"),
        "UV_PYTHON_INSTALL_BIN": "0",
        "PYTHONNOUSERSITE": "1",
        "KAGGLE_DISABLE_GOOGLE_GENERATIVE_AI_INTEGRATION": "1",
    })
    return environment


def python_version(python: Path, environment: dict) -> tuple[int, int] | None:
    """Inspect the executable, not the notebook version or a pyvenv.cfg label."""
    try:
        result = subprocess.run(
            [str(python), "-I", "-S", "-c",
             "import json, sys; print(json.dumps([sys.implementation.name, list(sys.version_info[:2])]))"],
            env=environment, check=True, capture_output=True, text=True, timeout=30,
        )
        implementation, version = json.loads(result.stdout)
        if implementation == "cpython" and isinstance(version, list) and len(version) == 2:
            return tuple(version)
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        pass
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_uv(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    archive_path = cache / f"uv-{UV_VERSION}.tar.gz"
    destination = cache / "uv"
    with tempfile.TemporaryDirectory(prefix="bootstrap-", dir=cache) as temporary:
        temporary = Path(temporary)
        if not archive_path.is_file():
            download = temporary / "uv.tar.gz"
            with urlopen(UV_URL, timeout=60) as response, download.open("wb") as output:
                shutil.copyfileobj(response, output)
            if _sha256(download) != UV_SHA256:
                raise RuntimeError("uv download checksum mismatch; no executable was installed.")
            download.replace(archive_path)
        if _sha256(archive_path) != UV_SHA256:
            raise RuntimeError("Cached uv archive checksum mismatch; use a fresh bootstrap cache directory.")
        # Extract only one regular file; never restore arbitrary archive paths.
        with tarfile.open(archive_path, "r:gz") as archive:
            member = archive.getmember(UV_MEMBER)
            if not member.isfile():
                raise RuntimeError("uv archive does not contain the expected executable.")
            binary = temporary / "uv"
            with archive.extractfile(member) as source, binary.open("wb") as output:
                shutil.copyfileobj(source, output)
        binary.chmod(0o755)
        binary.replace(destination)
    return destination


def ensure_python(cache: Path = DEFAULT_CACHE) -> Path:
    """Return CPython 3.12 using the specified bootstrap cache for downloads."""
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError("Python bootstrap targets Linux x86_64 Kaggle/Colab only.")
    cache = Path(cache).resolve()
    environment = bootstrap_environment(cache)
    for candidate in dict.fromkeys((sys.executable, shutil.which("python3.12"))):
        if candidate and python_version(Path(candidate), environment) == (3, 12):
            return Path(candidate)

    print("Preparing isolated Python 3.12; the notebook kernel remains unchanged.", flush=True)
    uv = _install_uv(cache)
    subprocess.run(
        [str(uv), "--no-config", "python", "install", "3.12", "--no-bin"],
        env=environment, check=True, timeout=600,
    )
    found = subprocess.run(
        [str(uv), "--no-config", "python", "find", "3.12", "--managed-python",
         "--no-python-downloads", "--system"],
        env=environment, check=True, capture_output=True, text=True, timeout=30,
    )
    python = Path(found.stdout.strip())
    if (not python.is_absolute() or not python.resolve().is_relative_to(cache / "python")
            or python_version(python, environment) != (3, 12)):
        raise RuntimeError("Managed Python 3.12 was not found after bootstrap; setup stopped.")
    return python
