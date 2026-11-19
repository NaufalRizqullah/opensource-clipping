"""Bootstrap regression tests to run in Kaggle; no real installs or processes."""

import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from scripts import kaggle_python as bootstrap
from scripts import kaggle_vllm as helper


@pytest.fixture(autouse=True)
def isolated_host(monkeypatch):
    monkeypatch.setattr(bootstrap.platform, "system", lambda: "Linux")
    monkeypatch.setattr(bootstrap.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(bootstrap.sys, "executable", "/kernel/python3.13")
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: None)

    def forbidden(*args, **kwargs):
        pytest.fail("Bootstrap tests must not start a process or download an artifact")

    monkeypatch.setattr(bootstrap.subprocess, "run", forbidden)
    monkeypatch.setattr(bootstrap, "urlopen", forbidden)


def test_bootstrap_keeps_secrets_but_isolates_python_uv_and_install_locations(monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap.os, "environ", {
        "PATH": "/usr/bin", "HF_TOKEN": "synthetic-token", "VLLM_API_KEY": "synthetic-key",
        "PYTHONHOME": "/kernel", "PYTHONPATH": "/kernel/packages", "VIRTUAL_ENV": "/other/venv",
        "UV_PYTHON_INSTALL_DIR": "/global", "UV_PYTHON_INSTALL_MIRROR": "https://untrusted.invalid",
        "PIP_TARGET": "/kernel/packages",
    })
    environment = bootstrap.bootstrap_environment(tmp_path)
    assert environment["HF_TOKEN"] == "synthetic-token"
    assert environment["VLLM_API_KEY"] == "synthetic-key"
    assert environment["UV_PYTHON_INSTALL_DIR"] == str(tmp_path / "python")
    assert environment["UV_PYTHON_INSTALL_BIN"] == "0"
    assert "UV_PYTHON_INSTALL_MIRROR" not in environment
    assert "PYTHONHOME" not in environment and "PYTHONPATH" not in environment
    assert "VIRTUAL_ENV" not in environment and "PIP_TARGET" not in environment
    assert bootstrap.os.environ["PYTHONHOME"] == "/kernel"


@pytest.mark.parametrize("kernel_is_312", [True, False])
def test_compatible_existing_python_skips_downloads(monkeypatch, tmp_path, kernel_is_312):
    selected = "/kernel/python3.13" if kernel_is_312 else "/usr/bin/python3.12"
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: "/usr/bin/python3.12")
    monkeypatch.setattr(bootstrap, "python_version", lambda path, env: (3, 12) if path == Path(selected) else (3, 13))
    assert bootstrap.ensure_python(tmp_path) == Path(selected)
    assert not (tmp_path / "uv").exists()


def test_python_313_kernel_installs_managed_312_and_finds_only_that_interpreter(monkeypatch, tmp_path):
    cache = tmp_path.resolve()
    selected = cache / "python" / "cpython-3.12-test" / "bin" / "python3.12"
    monkeypatch.setattr(bootstrap, "python_version", lambda path, env: (3, 12) if path == selected else (3, 13))
    monkeypatch.setattr(bootstrap, "_install_uv", lambda path: cache / "uv")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["check"] is True
        assert kwargs["env"]["UV_PYTHON_INSTALL_DIR"] == str(cache / "python")
        assert "--no-config" in command
        return SimpleNamespace(stdout=str(selected) + "\n")

    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    assert bootstrap.ensure_python(cache) == selected
    assert calls[0] == [str(cache / "uv"), "--no-config", "python", "install", "3.12", "--no-bin"]
    assert "--managed-python" in calls[1] and "--system" in calls[1]
    assert "--no-python-downloads" in calls[1]
    assert bootstrap.sys.executable == "/kernel/python3.13"


def test_failed_python_install_does_not_continue_to_discovery(monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap, "python_version", lambda *args: (3, 13))
    monkeypatch.setattr(bootstrap, "_install_uv", lambda path: path / "uv")
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(bootstrap.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        bootstrap.ensure_python(tmp_path)
    assert len(calls) == 1 and "install" in calls[0]


@pytest.mark.parametrize("outside_cache", [False, True])
def test_invalid_managed_result_is_rejected(monkeypatch, tmp_path, outside_cache):
    selected = tmp_path / ("outside" if outside_cache else "python") / "python3.12"
    monkeypatch.setattr(bootstrap, "python_version", lambda path, env: (3, 12) if outside_cache and path == selected else (3, 13))
    monkeypatch.setattr(bootstrap, "_install_uv", lambda path: path / "uv")
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=str(selected)))
    with pytest.raises(RuntimeError, match="Managed Python 3.12 was not found"):
        bootstrap.ensure_python(tmp_path)


@pytest.mark.parametrize("response,expected", [
    ('["cpython", [3, 12]]', (3, 12)), ('["cpython", [3, 13]]', (3, 13)),
    ('["pypy", [3, 12]]', None), ('unexpected banner', None), ('null', None),
])
def test_version_probe_checks_actual_interpreter(monkeypatch, response, expected):
    def probe(command, **kwargs):
        assert command[1:4] == ["-I", "-S", "-c"]
        assert kwargs["timeout"] == 30
        return SimpleNamespace(stdout=response)

    monkeypatch.setattr(bootstrap.subprocess, "run", probe)
    assert bootstrap.python_version(Path("/synthetic/python"), {}) == expected


def uv_archive(*, symlink=False):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        member = tarfile.TarInfo(bootstrap.UV_MEMBER)
        if symlink:
            member.type = tarfile.SYMTYPE
            member.linkname = "../../outside"
            archive.addfile(member)
        else:
            content = b"synthetic binary; never executed"
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        other = tarfile.TarInfo("../../must-not-extract")
        other.size = 1
        archive.addfile(other, io.BytesIO(b"x"))
    return stream.getvalue()


def test_uv_download_verifies_checksum_and_extracts_only_expected_file(monkeypatch, tmp_path):
    archive = uv_archive()
    monkeypatch.setattr(bootstrap, "UV_SHA256", hashlib.sha256(archive).hexdigest())
    monkeypatch.setattr(bootstrap, "urlopen", lambda *a, **kw: io.BytesIO(archive))
    executable = bootstrap._install_uv(tmp_path)
    assert executable.read_bytes() == b"synthetic binary; never executed"
    assert not (tmp_path.parent / "must-not-extract").exists()
    monkeypatch.setattr(bootstrap, "urlopen", lambda *a, **kw: pytest.fail("Reuse cached archive"))
    assert bootstrap._install_uv(tmp_path) == executable


def test_bad_download_never_installs_uv(monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap, "urlopen", lambda *a, **kw: io.BytesIO(b"wrong archive"))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        bootstrap._install_uv(tmp_path)
    assert not (tmp_path / "uv").exists()
    assert not (tmp_path / f"uv-{bootstrap.UV_VERSION}.tar.gz").exists()


def test_uv_archive_rejects_symlink_executable(monkeypatch, tmp_path):
    archive = uv_archive(symlink=True)
    monkeypatch.setattr(bootstrap, "UV_SHA256", hashlib.sha256(archive).hexdigest())
    monkeypatch.setattr(bootstrap, "urlopen", lambda *a, **kw: io.BytesIO(archive))
    with pytest.raises(RuntimeError, match="expected executable"):
        bootstrap._install_uv(tmp_path)
    assert not (tmp_path / "uv").exists()


def test_preflight_works_under_kernel_313_before_runtime_setup(monkeypatch):
    monkeypatch.setattr(helper.sys, "version_info", (3, 13, 0))

    def gpu_read(command, **kwargs):
        assert command[0] == "nvidia-smi"
        return SimpleNamespace(stdout="0, GPU-test, Tesla T4, 15360, 15000, 7.5, 580.00\n")

    monkeypatch.setattr(helper.subprocess, "run", gpu_read)
    assert helper.main(["preflight"]) == 0


def test_setup_under_kernel_313_has_actionable_error_before_creating_venv(monkeypatch, tmp_path):
    monkeypatch.setattr(helper.sys, "version_info", (3, 13, 0))
    with pytest.raises(RuntimeError, match="BOOTSTRAP_PYTHON"):
        helper.setup(tmp_path / "venv", helper.DEFAULT_VERSION)
    assert not (tmp_path / "venv").exists()


def test_control_commands_do_not_need_runtime_python(monkeypatch, tmp_path):
    from contextlib import nullcontext

    monkeypatch.setattr(helper.sys, "version_info", (3, 13, 0))
    monkeypatch.setattr(helper, "state_lock", lambda path: nullcontext())
    monkeypatch.setattr(helper, "read_state", lambda path: None)
    for command in ("status", "stop"):
        assert helper.main([command, "--state", str(tmp_path / "state.json")]) == 0


@pytest.mark.parametrize("filename", ["kaggle-vllm-gemma4.ipynb", "kaggle-studio-server.ipynb"])
def test_notebook_uses_selected_python_for_both_setup_helpers(filename):
    notebook = json.loads((bootstrap.ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
    calls = []

    def script_name(node):
        if isinstance(node, ast.Constant):
            return node.value
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "str" and len(node.args) == 1):
            path = node.args[0]
            if (isinstance(path, ast.BinOp) and isinstance(path.op, ast.Div)
                    and isinstance(path.left, ast.Name) and path.left.id == "PROJECT_DIR"
                    and isinstance(path.right, ast.Constant)):
                return path.right.value
        return None

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        assert cell.get("outputs", []) == []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call) or not node.args or not isinstance(node.args[0], ast.List):
                continue
            args = node.args[0].elts
            if len(args) < 3 or not isinstance(args[2], ast.Constant):
                continue
            helper_script = script_name(args[1])
            if helper_script not in {"scripts/kaggle_studio.py", "scripts/kaggle_vllm.py"} or args[2].value != "setup":
                continue
            assert ast.unparse(args[0]) == "str(BOOTSTRAP_PYTHON)"
            assert any(keyword.arg == "env" and ast.unparse(keyword.value) ==
                       "bootstrap_environment(PYTHON_CACHE) | cache_env" for keyword in node.keywords)
            calls.append(helper_script)
    assert sorted(calls) == ["scripts/kaggle_studio.py", "scripts/kaggle_vllm.py"]
