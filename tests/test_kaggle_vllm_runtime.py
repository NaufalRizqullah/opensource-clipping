"""Native-import regression checks with subprocesses replaced by fakes.

Run in the Kaggle application environment. These tests never install packages,
import vLLM/TorchCodec, contact a server, or load a model.
"""

from pathlib import Path
import re
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

import pytest

from scripts import kaggle_vllm as helper


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def isolated_server(monkeypatch, tmp_path):
    env_dir = tmp_path / "server-venv"
    env_dir.mkdir()
    environment = {"VIRTUAL_ENV": str(env_dir), "PATH": "isolated-server-path"}
    monkeypatch.setattr(helper, "ROOT", tmp_path)

    def runtime_environment(directory):
        assert directory == env_dir
        return environment

    monkeypatch.setattr(helper, "runtime_environment", runtime_environment)
    monkeypatch.setattr(helper.subprocess, "Popen", lambda *a, **kw: pytest.fail("No server may start"))
    return env_dir, environment


@pytest.mark.parametrize("failure", [None, "native_import", "timeout"])
def test_import_probe_uses_server_interpreter_and_preserves_diagnostics(
        monkeypatch, tmp_path, isolated_server, failure):
    env_dir, environment = isolated_server
    environment["VLLM_API_KEY"] = "private-server-key"
    calls = []
    diagnostic = "OSError: libnvrtc.so.13: cannot open shared object file\n"

    def run(command, **kwargs):
        calls.append(command)
        # --help exercises CLI imports without providing a model or serving port.
        assert command == [str(helper.environment_python(env_dir)), "-m",
                           "vllm.entrypoints.cli.main", "serve", "--help"]
        assert kwargs["cwd"] == tmp_path
        assert kwargs["env"] == {name: value for name, value in environment.items()
                                 if name != "VLLM_API_KEY"}
        assert kwargs["check"] is True and kwargs["timeout"] == 180
        assert kwargs["stdin"] == helper.subprocess.DEVNULL
        assert kwargs["stderr"] == helper.subprocess.STDOUT
        output = kwargs["stdout"]
        assert Path(output.name) == tmp_path / "logs/kaggle-vllm-import.log"
        message = diagnostic if failure else "vLLM serve usage\n"
        output.write(message.encode() if "b" in output.mode else message)
        if failure == "native_import":
            raise helper.subprocess.CalledProcessError(1, command)
        if failure == "timeout":
            raise helper.subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(helper.subprocess, "run", run)
    if failure:
        with pytest.raises(RuntimeError) as caught:
            helper.check_runtime_imports(env_dir)
        assert "import check" in str(caught.value).lower()
        assert str(tmp_path / "logs/kaggle-vllm-import.log") in str(caught.value)
        assert diagnostic in (tmp_path / "logs/kaggle-vllm-import.log").read_text()
    else:
        helper.check_runtime_imports(env_dir)
    assert len(calls) == 1
    assert environment["VLLM_API_KEY"] == "private-server-key"
    assert "private-server-key" not in (tmp_path / "logs/kaggle-vllm-import.log").read_text()


@pytest.mark.parametrize("failed_stage", [None, "install", "check", "imports"])
def test_setup_checks_native_imports_only_after_successful_dependency_checks(
        monkeypatch, tmp_path, isolated_server, failed_stage):
    env_dir, environment = isolated_server
    # Simulate the supported bootstrap interpreter without depending on the
    # notebook kernel's version or invoking a real virtualenv installer.
    monkeypatch.setattr(helper, "sys", SimpleNamespace(version_info=(3, 12, 0),
                                                      prefix=str(tmp_path / "kernel")))
    monkeypatch.setattr(helper, "require_platform", lambda: None)
    monkeypatch.setattr(helper, "validate_environment", lambda directory: None)
    monkeypatch.setattr(helper, "ensure_pip", lambda *args: None)
    manifest = (REPO_ROOT / "requirements-vllm-kaggle.txt").read_text(encoding="utf-8")
    (tmp_path / "requirements-vllm-kaggle.txt").write_text(manifest, encoding="utf-8")
    stages = []

    def run(command, **kwargs):
        assert command[:3] == [str(helper.environment_python(env_dir)), "-m", "pip"]
        assert kwargs["env"] is environment and kwargs["cwd"] == tmp_path
        assert kwargs["check"] is True
        stage = command[3]
        assert stage in {"install", "check"}
        stages.append(stage)
        if stage == "install":
            copied_manifest = Path(command[command.index("-r") + 1])
            assert copied_manifest.read_text(encoding="utf-8") == manifest
        if stage == failed_stage:
            raise helper.subprocess.CalledProcessError(1, command)
        return SimpleNamespace(returncode=0)

    def import_probe(directory):
        assert directory == env_dir
        stages.append("imports")
        if failed_stage == "imports":
            raise RuntimeError("Native import check failed")

    monkeypatch.setattr(helper.subprocess, "run", run)
    monkeypatch.setattr(helper, "check_runtime_imports", import_probe)
    if failed_stage:
        expected_error = RuntimeError if failed_stage == "imports" else helper.subprocess.CalledProcessError
        with pytest.raises(expected_error):
            helper.setup(env_dir, helper.DEFAULT_VERSION)
    else:
        helper.setup(env_dir, helper.DEFAULT_VERSION)
    expected = ["install", "check", "imports"]
    if failed_stage:
        expected = expected[:expected.index(failed_stage) + 1]
    assert stages == expected


def test_server_manifest_selects_cpu_codec_without_replacing_cuda_vllm():
    manifest = (REPO_ROOT / "requirements-vllm-kaggle.txt").read_text(encoding="utf-8")
    requirements = [line.strip() for line in manifest.splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
    codecs = [line for line in requirements if line.startswith("torchcodec @ ")]
    assert len(codecs) == 2
    python_versions = set()
    for requirement in codecs:
        url, marker = requirement.removeprefix("torchcodec @ ").split(";", 1)
        parsed = urlsplit(url.strip())
        assert parsed.scheme == "https" and parsed.netloc == "download.pytorch.org"
        wheel = unquote(parsed.path)
        assert wheel.startswith("/whl/cpu/torchcodec-0.17.0+cpu-")
        assert wheel.endswith("x86_64.whl")
        assert re.fullmatch(r"sha256=[0-9a-f]{64}", parsed.fragment)
        version = re.search(r"python_version\s*==\s*['\"](3\.1[12])['\"]", marker)
        assert version, "Each wheel must select its matching bootstrap Python version"
        python_versions.add(version.group(1))
        abi = "cp" + version.group(1).replace(".", "")
        assert f"-{abi}-{abi}-" in wheel
    assert python_versions == {"3.11", "3.12"}
    assert any(line.startswith("vllm @ ") and "vllm-0.32.0%2Bcu129" in line
               for line in requirements)
    assert "--extra-index-url https://download.pytorch.org/whl/cu129" in requirements
    assert not any(line.startswith(("--index-url", "--extra-index-url")) and "/cpu" in line
                   for line in requirements)
