"""Manage a single, loopback-only vLLM process in a separate Kaggle environment.

Only explicit setup/start commands install packages or load models. Imports use
the standard library and the existing Studio bootstrap; no ML stack is imported.
PID identity is checked against Linux process start time and a private owner token.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import math
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid
import venv

if __package__:
    from .kaggle_studio import ensure_pip, environment_python, runtime_environment, validate_environment
else:
    from kaggle_studio import ensure_pip, environment_python, runtime_environment, validate_environment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV = ROOT / ".cache" / "kaggle-vllm" / "venv"
DEFAULT_MODEL = "google/gemma-4-E2B-it-qat-w4a16-ct"
DEFAULT_REVISION = "971342c08f607aa7779983f6b5289778b5d271a7"
DEFAULT_VERSION = "0.32.0"
DEFAULT_STATE = ROOT / ".cache" / "kaggle-vllm" / "server.json"


def require_platform() -> None:
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError("This helper targets Linux x86_64 Kaggle/Colab only.")


def gpu_preflight(index: int = 0) -> dict:
    """Read NVIDIA metadata without importing Torch or allocating GPU memory."""
    require_platform()
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,compute_cap,driver_version",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True,
        timeout=30,
    )
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 7 or int(row[0].strip()) != index:
            continue
        info = {"index": index, "uuid": row[1].strip(), "name": row[2].strip(),
                "total_vram_mib": float(row[3]), "free_vram_mib": float(row[4]),
                "compute_capability": float(row[5]), "driver": row[6].strip()}
        if info["compute_capability"] < 7.5:
            raise RuntimeError("This vLLM/QAT profile requires SM 7.5 or newer. P100 (SM 6.0) is unsupported.")
        if info["total_vram_mib"] < 14 * 1024:
            raise RuntimeError("This profile targets a 16 GB-class GPU; less than 14 GiB was detected.")
        print(json.dumps(info, indent=2))
        print("VRAM is GPU memory; system RAM is separate. This check does not prove the model fits.")
        if info["compute_capability"] == 7.5:
            print("Turing GPU: start will check/apply the pinned Triton attention workaround before loading Gemma 4.")
        return info
    raise RuntimeError(f"NVIDIA GPU index {index} was not found.")


@contextmanager
def state_lock(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def read_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(result, dict) or type(result.get("pid")) is not int
                or result["pid"] <= 1 or not isinstance(result.get("identity"), list)
                or len(result["identity"]) != 2
                or not all(isinstance(value, str) for value in result["identity"])
                or not isinstance(result.get("owner"), str) or not result["owner"]
                or type(result.get("port")) is not int or not 1024 <= result["port"] <= 65535
                or not isinstance(result.get("model"), str)
                or not isinstance(result.get("log"), str)):
            raise ValueError
        return result
    except (OSError, ValueError):
        raise RuntimeError("Invalid vLLM state file; inspect it before restarting.") from None


def process_identity(pid: int) -> tuple[str, str] | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        # fields starts at proc stat field 3; starttime is field 22.
        if fields[0] == "Z":
            return None
        return fields[19], Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except FileNotFoundError:
        return None


def is_owned(state: dict) -> bool:
    identity = process_identity(state["pid"])
    if identity is None:
        return False
    if list(identity) != state.get("identity"):
        raise RuntimeError("Saved PID belongs to another process; refusing to signal or adopt it.")
    marker = ("OSC_VLLM_OWNER=" + state["owner"]).encode()
    try:
        environment = Path(f"/proc/{state['pid']}/environ").read_bytes().split(b"\0")
        group = os.getpgid(state["pid"])
    except (FileNotFoundError, ProcessLookupError):
        return False
    if marker not in environment or group != state["pid"]:
        raise RuntimeError("vLLM process ownership could not be verified; refusing to signal it.")
    return True


def write_state(path: Path, state: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix="server-", suffix=".json", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(state, handle)
    temporary.chmod(0o600)
    temporary.replace(path)


def request_json(url: str) -> dict:
    headers = {}
    key = os.environ.get("VLLM_API_KEY", "")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    with urlopen(Request(url, headers=headers), timeout=5) as response:
        return json.load(response)


def wait_ready(state: dict, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_owned(state):
            raise RuntimeError(f"vLLM exited before readiness; inspect {state['log']}.")
        try:
            models = request_json(f"http://127.0.0.1:{state['port']}/v1/models")
            available = models.get("data", []) if isinstance(models, dict) else []
            if isinstance(available, list) and any(
                    isinstance(item, dict) and item.get("id") == state["model"] for item in available):
                print(f"vLLM is ready: http://127.0.0.1:{state['port']}/v1")
                return
        except HTTPError as exc:
            if exc.code in {401, 403}:
                raise RuntimeError("vLLM authentication failed. Restore the original VLLM_API_KEY or stop/restart.") from None
        except (URLError, TimeoutError, OSError, ValueError):
            pass
        time.sleep(2)
    raise RuntimeError(f"vLLM readiness timed out; inspect {state['log']}.")


def stop_owned(path: Path, state: dict, timeout: float = 30) -> None:
    if is_owned(state):
        os.killpg(state["pid"], signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while is_owned(state) and time.monotonic() < deadline:
            time.sleep(0.5)
        if is_owned(state):
            os.killpg(state["pid"], signal.SIGKILL)
    path.unlink(missing_ok=True)
    print("Managed vLLM server stopped.")


def setup(env_dir: Path, version: str) -> None:
    require_platform()
    if not (3, 11) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError(
            f"Setup is running under Python {sys.version_info[0]}.{sys.version_info[1]}. "
            "Run the notebook's Prepare Python cell, then use BOOTSTRAP_PYTHON for setup. "
            "The model/application environments require Python 3.11 or 3.12; the kernel may be newer."
        )
    if env_dir.resolve() in {(ROOT / ".cache/kaggle-studio/venv").resolve(), Path(sys.prefix).resolve()}:
        raise RuntimeError("Use a separate vLLM environment, never the Studio environment.")
    if version != DEFAULT_VERSION:
        raise RuntimeError("This profile pins vLLM and Transformers together; update requirements-vllm-kaggle.txt to change versions.")
    if not env_dir.exists():
        venv.EnvBuilder(with_pip=False, system_site_packages=False).create(env_dir)
    validate_environment(env_dir)
    ensure_pip(env_dir, ROOT)
    requirements = (ROOT / "requirements-vllm-kaggle.txt").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="clipping-vllm-") as directory:
        target = Path(directory) / "requirements.txt"
        target.write_text(requirements, encoding="utf-8")
        python = str(environment_python(env_dir))
        environment = runtime_environment(env_dir)
        subprocess.run([python, "-m", "pip", "install", "--upgrade", "-r", str(target),
                        "--index-url", "https://pypi.org/simple"],
                       cwd=ROOT, env=environment, check=True)
        subprocess.run([python, "-m", "pip", "check"], cwd=ROOT, env=environment, check=True)
    print("vLLM dependencies installed in isolation. GPU/model startup remains unverified until start.")


def build_server_command(args) -> list[str]:
    """Build the Gemma 4 text-only profile without importing or starting vLLM."""
    revision = args.revision
    if revision is None and args.model == DEFAULT_MODEL:
        revision = DEFAULT_REVISION
    command = [str(environment_python(args.venv)), "-m", "vllm.entrypoints.cli.main", "serve", args.model,
               "--host", "127.0.0.1", "--port", str(args.port),
               "--dtype", "half", "--tensor-parallel-size", "1",
               "--max-model-len", str(args.context), "--max-num-seqs", str(args.max_num_seqs),
               "--gpu-memory-utilization", str(args.gpu_memory_utilization),
               "--enforce-eager", "--language-model-only", "--reasoning-parser", "gemma4",
               "--no-enable-log-requests", "--no-enable-log-outputs",
               "--default-chat-template-kwargs", '{"enable_thinking":false}']
    if revision:
        command += ["--revision", revision, "--tokenizer-revision", revision]
    # The checkpoint declares compressed-tensors W4A16. Do not substitute a
    # runtime quantizer or download a different, unquantized checkpoint.
    return command


def prepare_attention_compat(env_dir: Path, gpu: dict, environment: dict) -> None:
    """Patch only the server venv on Turing; no GPU work occurs in this process."""
    if gpu["compute_capability"] != 7.5:
        return
    subprocess.run(
        [str(environment_python(env_dir)), str(ROOT / "scripts/kaggle_vllm_compat.py"),
         "--venv", str(env_dir)],
        cwd=ROOT, env=environment, check=True, timeout=30,
    )


def start(args) -> None:
    require_platform()
    validate_environment(args.venv)
    command = build_server_command(args)
    existing = read_state(args.state)
    if existing and is_owned(existing):
        if existing.get("command") != command or existing.get("gpu_index") != args.gpu_index:
            raise RuntimeError("A managed server has different settings. Run stop before changing configuration.")
        wait_ready(existing, args.startup_timeout)
        return
    # Never adopt a process merely because a port answers HTTP.
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError:
            raise RuntimeError(f"Port {args.port} is busy; stop its owner or choose another port.") from None
    gpu = gpu_preflight(args.gpu_index)
    environment = runtime_environment(args.venv)
    owner = uuid.uuid4().hex
    environment.update({"CUDA_VISIBLE_DEVICES": gpu["uuid"], "OSC_VLLM_OWNER": owner,
                        "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
                        "VLLM_LOGGING_LEVEL": "INFO", "VLLM_DEBUG_LOG_API_SERVER_RESPONSE": "False"})
    # This also repairs an already-installed server env after a checkout update.
    # A source/version mismatch must fail before Popen, never after model loading.
    prepare_attention_compat(args.venv, gpu, environment)
    log = ROOT / "logs" / "kaggle-vllm.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as output:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    identity = process_identity(process.pid)
    if identity is None:
        raise RuntimeError(f"vLLM exited immediately; inspect {log}.")
    state = {"pid": process.pid, "identity": list(identity), "owner": owner,
             "command": command, "port": args.port, "model": args.model,
             "gpu_index": args.gpu_index, "log": str(log)}
    try:
        write_state(args.state, state)
        wait_ready(state, args.startup_timeout)
    except BaseException:
        stop_owned(args.state, state)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "setup", "start", "status", "stop"))
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--vllm-version", default=DEFAULT_VERSION)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=None, help="HF revision; default checkpoint is commit-pinned")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--startup-timeout", type=float, default=1200)
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535 or args.context <= 0 or args.max_num_seqs <= 0:
        parser.error("Port must be 1024-65535; context and max-num-seqs must be positive.")
    if (not 0 < args.gpu_memory_utilization < 1 or not math.isfinite(args.startup_timeout)
            or args.startup_timeout <= 0 or args.gpu_index < 0):
        parser.error("Invalid GPU memory fraction, startup timeout, or GPU index.")
    require_platform()
    if args.command == "preflight":
        gpu_preflight(args.gpu_index)
        return 0
    with state_lock(args.state):
        if args.command == "setup":
            state = read_state(args.state)
            if state and is_owned(state):
                raise RuntimeError("Stop the managed server before modifying its environment.")
            setup(args.venv, args.vllm_version)
        elif args.command == "start":
            start(args)
        else:
            state = read_state(args.state)
            if args.command == "stop" and state:
                stop_owned(args.state, state)
            elif not state or not is_owned(state):
                print("No live managed vLLM server.")
            else:
                print(f"Managed vLLM PID {state['pid']} on loopback port {state['port']}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"vLLM helper: {error}", file=sys.stderr)
        raise SystemExit(1)
