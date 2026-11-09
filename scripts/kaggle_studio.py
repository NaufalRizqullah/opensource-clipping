"""Isolated Kaggle/Colab setup and supervised Web Studio startup.

This module uses only the standard library until the tunnel is opened. It can
therefore run with the notebook's Python without importing its NumPy/ML stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import venv
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV = PROJECT_ROOT / ".cache" / "kaggle-studio" / "venv"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
# Official release asset and its published SHA-256 (denoland/deno v2.9.8).
DENO_VERSION = "2.9.8"
DENO_SHA256 = "e890975bc2549807bbcc0b584ffafda72cc7c93349dd99c80483dd784f6c4609"
DENO_URL = (
    f"https://github.com/denoland/deno/releases/download/v{DENO_VERSION}/"
    "deno-x86_64-unknown-linux-gnu.zip"
)


def environment_python(env_dir: Path) -> Path:
    """Return the interpreter inside the managed environment."""
    return env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def runtime_environment(env_dir: Path) -> dict[str, str]:
    """Keep API secrets in memory while excluding notebook Python overrides."""
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        environment.pop(name, None)
    # A notebook/user pip config must not redirect installation outside the venv.
    for name in list(environment):
        if name.startswith("PIP_"):
            environment.pop(name)
    environment.update({
        "VIRTUAL_ENV": str(env_dir),
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHON_DOTENV_DISABLED": "1",
        # Once wrapt is present, Kaggle's sitecustomize can patch google.genai.
        # Keep Studio's explicit API-key/provider routing instead of its proxy.
        "KAGGLE_DISABLE_GOOGLE_GENERATIVE_AI_INTEGRATION": "1",
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_REQUIRE_VIRTUALENV": "true",
    })
    bin_dir = environment_python(env_dir).parent
    environment["PATH"] = str(bin_dir) + os.pathsep + environment.get("PATH", "")
    # CTranslate2 needs the CUDA libraries installed by the matched torch wheel.
    cuda_dirs = sorted(env_dir.glob("lib/python*/site-packages/nvidia/*/lib"))
    if cuda_dirs:
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            [*(str(path) for path in cuda_dirs), environment.get("LD_LIBRARY_PATH", "")]
        ).rstrip(os.pathsep)
    return environment


def validate_platform() -> None:
    """Reject unsupported Python/wheel platforms before any installation."""
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError("Kaggle Studio requires Python 3.10-3.12; select a Python 3.12 runtime.")
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError("This bootstrap targets Linux x86_64 Kaggle/Colab runtimes.")


def validate_environment(env_dir: Path) -> None:
    """Do not reuse an environment that can see the notebook's site packages."""
    config_path = env_dir / "pyvenv.cfg"
    if not config_path.is_file():
        raise RuntimeError("Managed environment is missing. Run the setup cell first.")
    config = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            config[key.strip().lower()] = value.strip().lower()
    if config.get("include-system-site-packages") != "false":
        raise RuntimeError("Environment is not isolated. Choose a new directory with --venv.")
    if not environment_python(env_dir).is_file():
        raise RuntimeError("Environment interpreter is missing. Choose a new directory with --venv.")


def install_deno(env_dir: Path) -> None:
    """Install the verified official binary used by yt-dlp's JS challenge solver."""
    destination = environment_python(env_dir).parent / "deno"
    if destination.is_file():
        result = subprocess.run(
            [str(destination), "--version"], capture_output=True, text=True, check=True,
        )
        if result.stdout.splitlines()[0] == f"deno {DENO_VERSION} (stable, release, x86_64-unknown-linux-gnu)":
            return
        # The version's first line has changed format between Deno releases.
        if result.stdout.startswith(f"deno {DENO_VERSION}\n"):
            return
    print(f"Installing Deno {DENO_VERSION} for YouTube downloads...", flush=True)
    with tempfile.TemporaryDirectory(prefix="clipping-deno-") as temporary:
        archive_path = Path(temporary) / "deno.zip"
        digest = hashlib.sha256()
        with urlopen(DENO_URL, timeout=60) as response, archive_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != DENO_SHA256:
            raise RuntimeError("Deno download checksum mismatch; no binary was installed.")
        # Extract only the named binary, never arbitrary archive paths.
        with zipfile.ZipFile(archive_path) as archive, archive.open("deno") as source:
            temporary_binary = Path(temporary) / "deno"
            with temporary_binary.open("wb") as target:
                shutil.copyfileobj(source, target)
        temporary_binary.chmod(0o755)
        shutil.copy2(temporary_binary, destination)
    subprocess.run([str(destination), "--version"], check=True)


def ensure_pip(env_dir: Path, root: Path = PROJECT_ROOT) -> None:
    """Repair pip inside a valid venv without depending on distro ensurepip."""
    python = str(environment_python(env_dir))
    environment = runtime_environment(env_dir)
    # A pyvenv.cfg file alone does not prove this interpreter uses that venv.
    try:
        probe = subprocess.run(
            [python, "-I", "-c", "import json, sys; print(json.dumps([sys.prefix, sys.base_prefix]))"],
            cwd=root, env=environment, capture_output=True, text=True, timeout=30, check=True,
        )
        prefix, base_prefix = json.loads(probe.stdout)
        if Path(prefix).resolve() != env_dir.resolve() or Path(prefix).resolve() == Path(base_prefix).resolve():
            raise ValueError("Interpreter is not isolated")
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        raise RuntimeError("Managed Python is not using the isolated environment; choose a new --venv directory.") from None

    command = [python, "-I", "-m", "pip", "--version"]
    if subprocess.run(
        command, cwd=root, env=environment, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, timeout=30, check=False,
    ).returncode == 0:
        return

    print("Bootstrapping pip inside the isolated environment...", flush=True)
    # pip >=22.3 can manage a different Python even when it has no pip yet.
    # This installs into the target, never into the notebook/kernel interpreter.
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "--python", python, "install", "--upgrade", "pip",
             "--index-url", "https://pypi.org/simple"],
            cwd=root, env=environment, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        print("Host pip unavailable; using the official PyPA bootstrap in the target environment...", flush=True)
        with tempfile.TemporaryDirectory(prefix="clipping-pip-") as temporary:
            script = Path(temporary) / "get-pip.py"
            try:
                with urlopen(GET_PIP_URL, timeout=60) as response, script.open("wb") as output:
                    shutil.copyfileobj(response, output)
            except OSError:
                raise RuntimeError("Could not download the PyPA pip bootstrap. Check notebook Internet access and rerun setup.") from None
            subprocess.run(
                [python, "-I", str(script), "--no-setuptools", "--no-wheel",
                 "--index-url", "https://pypi.org/simple"],
                cwd=root, env=environment, check=True,
            )
    subprocess.run(command, cwd=root, env=environment, timeout=30, check=True)


def setup_environment(env_dir: Path, root: Path = PROJECT_ROOT) -> None:
    """Install only into an isolated venv; any failed command stops setup."""
    validate_platform()
    if not env_dir.exists():
        print("Creating isolated Kaggle Studio environment...", flush=True)
        venv.EnvBuilder(with_pip=False, system_site_packages=False).create(env_dir)
    validate_environment(env_dir)
    ensure_pip(env_dir, root)
    python = str(environment_python(env_dir))
    environment = runtime_environment(env_dir)
    for command in (
        [python, "-m", "pip", "install", "--upgrade", "pip", "--index-url", "https://pypi.org/simple"],
        [python, "-m", "pip", "install", "--upgrade", "-r", str(root / "requirements-kaggle.txt"),
         "--index-url", "https://pypi.org/simple"],
        [python, "-m", "pip", "check"],
    ):
        subprocess.run(command, cwd=root, env=environment, check=True)
    install_deno(env_dir)
    print("Dependencies installed. The start cell will verify actual renderer imports.", flush=True)


def wait_for_readiness(process, port: int, timeout: float = 300) -> dict:
    """Require a ready renderer, not merely an open port or liveness response."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"FastAPI exited during startup (exit code {process.returncode}).")
        remaining = deadline - time.monotonic()
        try:
            # The first request performs the cached import probe (up to 120s).
            with urlopen(f"http://127.0.0.1:{port}/api/ready", timeout=min(150, remaining)) as response:
                report = json.load(response)
            if report.get("ready") is True and process.poll() is None:
                return report
            raise RuntimeError("FastAPI did not pass its renderer readiness checks.")
        except HTTPError as error:
            if error.code == 503:
                try:
                    report = json.load(error)
                except (ValueError, OSError):
                    report = {}
                # Print only known check names/classes, not arbitrary response text.
                failures = []
                for name in ("numpy", "renderer", "transcription", "ffmpeg", "runtime"):
                    check = report.get("checks", {}).get(name, {})
                    if check.get("ok") is False:
                        kind = check.get("error")
                        if kind not in {"ValueError", "ImportError", "ModuleNotFoundError", "RuntimeError", "OSError", "AttributeError"}:
                            kind = "unavailable"
                        failures.append(f"{name}: {kind}")
                raise RuntimeError(
                    "Renderer readiness failed (" + (", ".join(failures) or "runtime unavailable")
                    + "). Recreate the isolated environment and rerun setup. "
                    "No public tunnel was opened."
                ) from None
            raise RuntimeError(f"Unexpected readiness HTTP status {error.code}; update the backend checkout.") from None
        except (URLError, TimeoutError, OSError):
            time.sleep(min(1, max(0, deadline - time.monotonic())))
    raise RuntimeError("Timed out waiting for FastAPI renderer readiness; no public tunnel was opened.")


def stop_process(process) -> None:
    """Stop the supervised server without deleting its jobs or outputs."""
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def open_tunnel(port: int, env_dir: Path):
    """Open ngrok only after readiness; auth stays in the inherited environment."""
    from pyngrok import conf, ngrok

    logging.getLogger("pyngrok").setLevel(logging.CRITICAL)
    config = conf.PyngrokConfig(
        ngrok_path=str(env_dir.parent / "ngrok"),
        config_path=str(env_dir.parent / "ngrok.yml"),
        start_new_session=True,
    )
    # ngrok itself consumes NGROK_AUTHTOKEN; avoid --authtoken argv or config writes.
    config.auth_token = None
    try:
        tunnel = ngrok.connect(addr=f"127.0.0.1:{port}", proto="http", bind_tls=True, pyngrok_config=config)
    except Exception:
        ngrok.kill(pyngrok_config=config)
        raise RuntimeError("ngrok failed to open a tunnel. Check NGROK_AUTHTOKEN and notebook internet access.") from None
    return tunnel, lambda: ngrok.kill(pyngrok_config=config)


def start_server(env_dir: Path, root: Path = PROJECT_ROOT, port: int = 8000,
                 readiness_timeout: float = 300) -> None:
    """Launch FastAPI in a fresh interpreter, verify imports, then publish its URL."""
    validate_environment(env_dir)
    if not os.environ.get("NGROK_AUTHTOKEN", "").strip():
        raise RuntimeError(
            "NGROK_AUTHTOKEN is missing from the server environment. "
            "Attach the secret with this exact label to the notebook, then rerun the Configure API Keys cell."
        )
    # On Linux the venv executable often symlinks to the system interpreter;
    # comparing resolved executable paths cannot prove the venv is active.
    if Path(sys.prefix).resolve() != env_dir.resolve():
        raise RuntimeError("Run the start command with the isolated environment's Python.")
    # Never mistake a server left over from another cell for this new process.
    with socket.socket() as candidate:
        try:
            candidate.bind(("127.0.0.1", port))
        except OSError:
            raise RuntimeError(f"Port {port} is already in use. Stop the earlier server cell first.") from None
    process = subprocess.Popen(
        [str(environment_python(env_dir)), "-m", "uvicorn", "web.api.app:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "info"],
        cwd=root, env=runtime_environment(env_dir),
    )
    close_tunnel = None
    try:
        print("Waiting for FastAPI and renderer readiness...", flush=True)
        wait_for_readiness(process, port, readiness_timeout)
        tunnel, close_tunnel = open_tunnel(port, env_dir)
        print(f"\nBACKEND SERVER READY\nPublic URL: {tunnel.public_url}", flush=True)
        print("Paste this URL into Studio > Connect. Keep this cell running.", flush=True)
        result = process.wait()
        if result not in (0, -signal.SIGINT):
            raise RuntimeError(f"FastAPI stopped with exit code {result}; inspect the server logs above.")
    finally:
        try:
            if close_tunnel is not None:
                close_tunnel()
        finally:
            stop_process(process)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("setup", "start"))
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--readiness-timeout", type=float, default=300)
    args = parser.parse_args(argv)
    env_dir = args.venv.resolve()
    # A stopped notebook cell/terminated supervisor must also stop the backend.
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.command == "setup":
            setup_environment(env_dir)
        else:
            start_server(env_dir, port=args.port, readiness_timeout=args.readiness_timeout)
    except KeyboardInterrupt:
        print("Studio server stopped.", flush=True)
        return 130
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Studio startup failed: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
