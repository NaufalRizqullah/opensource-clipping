"""Notebook wiring regressions for execution in Kaggle, never on the user's PC.

Only individual source cells run under fakes; no installer, model, GPU, network
service or real subprocess is started by these tests.
"""

import ast
import json
from pathlib import Path
import secrets
import subprocess
from types import SimpleNamespace

import pytest

from scripts import kaggle_python
from scripts import kaggle_storage
from scripts import kaggle_studio


def cell_source(*markers):
    notebook = json.loads((kaggle_python.ROOT / "notebooks/kaggle-studio-server.ipynb").read_text(encoding="utf-8"))
    sources = [cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
               for cell in notebook["cells"] if cell["cell_type"] == "code"]
    candidates = [source for source in sources if all(marker in source for marker in markers)]
    assert len(candidates) == 1, f"Expected exactly one notebook cell matching {markers}"
    return candidates[0]


@pytest.fixture
def notebook_context(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Notebook tests must not start a real process or download dependencies")

    monkeypatch.setattr(kaggle_python.subprocess, "run", forbidden)
    monkeypatch.setattr(kaggle_python.subprocess, "Popen", forbidden)
    monkeypatch.setattr(kaggle_python, "urlopen", forbidden)
    monkeypatch.setattr(kaggle_studio, "urlopen", forbidden)
    monkeypatch.setattr(kaggle_python.os, "environ", {})
    workspace = tmp_path / "working"
    project = workspace / "checkout"
    project.mkdir(parents=True)
    storage = kaggle_storage.prepare_storage(project, workspace, runtime_root=tmp_path / "scratch")
    monkeypatch.setattr(kaggle_storage, "prepare_storage", lambda *args, **kwargs: storage)
    monkeypatch.setattr(kaggle_storage, "check_storage", lambda *args: None)
    return {
        "PROJECT_DIR": project, "workspace": workspace,
        "Path": Path, "os": kaggle_python.os,
        "sys": SimpleNamespace(executable="/kernel/python3.13",
                               version_info=SimpleNamespace(major=3, minor=13)),
        "subprocess": subprocess,
        "BOOTSTRAP_PYTHON": tmp_path / "python3.12",
        "storage": storage, "MIN_OUTPUT_GIB": 1.0, "STUDIO_TORCH_PROFILE": "cuda",
        "cache_env": storage.cache_environment(), "check_storage": kaggle_storage.check_storage,
        "PYTHON_CACHE": storage.python_cache, "STUDIO_ENV": storage.studio_env,
        "STUDIO_PYTHON": storage.studio_env / "bin/python",
        "bootstrap_environment": kaggle_python.bootstrap_environment,
        "runtime_environment": kaggle_studio.runtime_environment,
    }


def test_prepare_python_reuses_shared_bootstrap_and_preserves_kernel(monkeypatch, notebook_context):
    context = notebook_context
    selected = context["BOOTSTRAP_PYTHON"]
    requests = []

    def ensure_python(cache):
        requests.append(cache)
        return selected

    def run(command, **kwargs):
        assert command == [str(selected), "--version"]
        assert kwargs["check"] is True
        assert kwargs["env"]["UV_PYTHON_INSTALL_DIR"] == str(context["PYTHON_CACHE"] / "python")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(kaggle_python, "ensure_python", ensure_python)
    monkeypatch.setattr(subprocess, "run", run)
    exec(compile(cell_source("ensure_python(PYTHON_CACHE)"), "studio-prepare-python", "exec"), context)
    assert requests == [context["PYTHON_CACHE"]]
    assert context["BOOTSTRAP_PYTHON"] == selected
    assert context["sys"].executable == "/kernel/python3.13"


@pytest.mark.parametrize("profile", ["cuda", "cpu"])
def test_studio_setup_uses_selected_python_with_fresh_isolated_environment(monkeypatch, notebook_context, profile):
    context = notebook_context
    context["STUDIO_TORCH_PROFILE"] = profile
    monkeypatch.setenv("NGROK_AUTHTOKEN", "late-secret-value")
    monkeypatch.setenv("PYTHONPATH", "/kernel/packages")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[0] == str(context["BOOTSTRAP_PYTHON"])
        assert str(command[1]).replace("\\", "/").endswith("scripts/kaggle_studio.py")
        assert command[2] == "setup"
        assert command[command.index("--venv") + 1] == str(context["STUDIO_ENV"])
        assert command[command.index("--torch-profile") + 1] == profile
        assert context["STUDIO_ENV"].is_relative_to(context["storage"].root)
        assert context["STUDIO_ENV"].parent.name == ("kaggle-studio-cpu" if profile == "cpu" else "kaggle-studio")
        assert kwargs["env"]["TMPDIR"] == str(context["storage"].root / "tmp")
        assert kwargs["env"]["NGROK_AUTHTOKEN"] == "late-secret-value"
        assert "PYTHONPATH" not in kwargs["env"]
        assert kwargs["check"] is True
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    exec(compile(cell_source("kaggle_studio.py", '"setup"'), "studio-install", "exec"), context)
    assert len(calls) == 1


def test_insufficient_space_prevents_backend_start(monkeypatch, notebook_context):
    def insufficient(layout, reserve):
        raise RuntimeError("Insufficient output space")

    notebook_context["check_storage"] = insufficient
    with pytest.raises(RuntimeError, match="Insufficient output space"):
        exec(compile(cell_source("supervisor = subprocess.Popen"), "studio-start", "exec"), notebook_context)


def configured_source(local):
    """Model the notebook's user-editable switch without changing execution code."""
    tree = ast.parse(cell_source("START_LOCAL_VLLM ="))
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "START_LOCAL_VLLM"
                           for target in node.targets)]
    assert len(assignments) == 1 and isinstance(assignments[0].value, ast.Constant)
    assert assignments[0].value.value is False, "Local model installation must remain opt-in"
    assignments[0].value = ast.copy_location(ast.Constant(value=local), assignments[0].value)
    return compile(ast.fix_missing_locations(tree), "studio-provider-configuration", "exec")


def test_attached_cookie_file_is_referenced_in_place(monkeypatch, tmp_path, notebook_context, capsys):
    source = tmp_path / "private-input" / "cookies.txt"
    source.parent.mkdir()
    source.write_text("# synthetic notebook fixture")
    project = notebook_context["PROJECT_DIR"]
    monkeypatch.chdir(project)
    monkeypatch.setenv("OSC_YOUTUBE_COOKIES_FILE", str(source))
    exec(compile(cell_source("cookie_candidates ="), "studio-cookie-input", "exec"), notebook_context)
    assert notebook_context["os"].environ["OSC_YOUTUBE_COOKIES_FILE"] == str(source)
    assert not (project / "youtube_cookies.txt").exists()
    assert "synthetic notebook fixture" not in capsys.readouterr().out


def run_local_server_cell(context):
    source = cell_source("kaggle_vllm_compat.py", '"setup"', '"preflight"')
    exec(compile(source, "studio-local-vllm", "exec"), context)


def run_stop_cell(context):
    exec(compile(cell_source("START_LOCAL_VLLM", '"stop"'), "studio-stop-vllm", "exec"), context)


def prepare_local_files(context):
    for filename in ("scripts/kaggle_vllm.py", "scripts/kaggle_vllm_compat.py",
                     "scripts/eval_local_llm.py", "requirements-vllm-kaggle.txt"):
        path = context["PROJECT_DIR"] / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Fake fixture; never executed\n", encoding="utf-8")


def test_default_cloud_mode_does_not_install_start_evaluate_or_stop_vllm(
    monkeypatch, notebook_context
):
    monkeypatch.setattr(secrets, "token_urlsafe", lambda *a: pytest.fail("No local key for cloud mode"))
    exec(configured_source(False), notebook_context)
    assert "VLLM_API_KEY" not in kaggle_python.os.environ
    run_local_server_cell(notebook_context)
    run_stop_cell(notebook_context)
    # The fixture forbids all subprocesses; even querying a GPU would fail.


def test_existing_vllm_endpoint_and_key_survive_external_mode(monkeypatch, notebook_context, capsys):
    endpoint = "https://private-server.example/v1"
    monkeypatch.setenv("VLLM_BASE_URL", endpoint)
    monkeypatch.setenv("VLLM_MODEL", "operator/model-alias")
    monkeypatch.setenv("VLLM_API_KEY", "existing-server-key")
    monkeypatch.setattr(secrets, "token_urlsafe", lambda *a: pytest.fail("Do not replace an external key"))
    exec(configured_source(False), notebook_context)
    run_local_server_cell(notebook_context)
    environment = kaggle_studio.runtime_environment(notebook_context["STUDIO_ENV"])
    assert environment["VLLM_BASE_URL"] == endpoint
    assert environment["VLLM_MODEL"] == "operator/model-alias"
    assert environment["VLLM_API_KEY"] == "existing-server-key"
    assert "existing-server-key" not in capsys.readouterr().out


@pytest.mark.parametrize("existing_key", [None, "existing-local-key"])
def test_opt_in_config_uses_loopback_and_retains_secret_between_reruns(
    monkeypatch, notebook_context, capsys, existing_key
):
    monkeypatch.setenv("VLLM_BASE_URL", "https://old-remote.example/v1")
    if existing_key:
        monkeypatch.setenv("VLLM_API_KEY", existing_key)
    generated = []

    def token_urlsafe(length):
        generated.append(length)
        return "generated-test-key"

    monkeypatch.setattr(secrets, "token_urlsafe", token_urlsafe)
    exec(configured_source(True), notebook_context)
    exec(configured_source(True), notebook_context)
    assert kaggle_python.os.environ["VLLM_BASE_URL"] == "http://127.0.0.1:8001"
    assert kaggle_python.os.environ["VLLM_API_KEY"] == (existing_key or "generated-test-key")
    assert generated == ([] if existing_key else [32])
    output = capsys.readouterr().out
    assert "generated-test-key" not in output and "existing-local-key" not in output


def test_local_setup_start_and_eval_use_separate_interpreters(monkeypatch, notebook_context):
    context = notebook_context
    exec(configured_source(True), context)
    prepare_local_files(context)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["cwd"] == context["PROJECT_DIR"]
        assert kwargs["env"]["VLLM_API_KEY"] == kaggle_python.os.environ["VLLM_API_KEY"]
        if "scripts/eval_local_llm.py" in command:
            assert command[0] == str(context["STUDIO_PYTHON"])
            assert kwargs["env"]["VIRTUAL_ENV"] == str(context["STUDIO_ENV"])
            assert kwargs["check"] is False
        else:
            assert command[0] == str(context["BOOTSTRAP_PYTHON"])
            assert kwargs["check"] is True
            assert "VIRTUAL_ENV" not in kwargs["env"]
            if command[2] in {"setup", "start"}:
                assert command[command.index("--venv") + 1] == str(context["VLLM_ENV"])
                assert context["VLLM_ENV"] != context["STUDIO_ENV"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    run_local_server_cell(context)
    assert [command[2] for command in calls[:3]] == ["preflight", "setup", "start"]
    assert len(calls) == 4
    assert calls[3][1] == "scripts/eval_local_llm.py"


def test_failed_gpu_preflight_prevents_model_installation(monkeypatch, notebook_context):
    context = notebook_context
    exec(configured_source(True), context)
    prepare_local_files(context)
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        run_local_server_cell(context)
    assert len(calls) == 1 and calls[0][2] == "preflight"


def test_failed_eval_displays_server_log_and_remains_a_failure(monkeypatch, notebook_context, capsys):
    context = notebook_context
    exec(configured_source(True), context)
    prepare_local_files(context)
    log = context["PROJECT_DIR"] / "logs/kaggle-vllm.log"
    log.parent.mkdir()
    log.write_text("Synthetic kernel failure for diagnosis\n", encoding="utf-8")
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: SimpleNamespace(
        returncode=1 if "scripts/eval_local_llm.py" in command else 0,
    ))
    with pytest.raises(RuntimeError, match="vLLM eval failed"):
        run_local_server_cell(context)
    assert "Synthetic kernel failure" in capsys.readouterr().out


def test_local_stop_uses_managed_helper_without_changing_python(monkeypatch, notebook_context):
    context = notebook_context
    context["START_LOCAL_VLLM"] = True
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[0] == context["sys"].executable
        assert str(command[1]).replace("\\", "/").endswith("scripts/kaggle_vllm.py")
        assert command[2:] == ["stop"]
        assert kwargs["check"] is True

    monkeypatch.setattr(subprocess, "run", run)
    run_stop_cell(context)
    assert len(calls) == 1


def test_studio_supervisor_receives_vllm_configuration_and_cleans_up(monkeypatch, notebook_context):
    context = notebook_context
    monkeypatch.setenv("VLLM_BASE_URL", "https://existing.example/v1")
    monkeypatch.setenv("VLLM_API_KEY", "matching-server-key")
    monkeypatch.setenv("NGROK_AUTHTOKEN", "tunnel-key")
    exec(configured_source(False), context)
    child = SimpleNamespace(wait=lambda: 0)
    stopped = []

    def spawn(command, **kwargs):
        assert command[0] == str(context["STUDIO_PYTHON"])
        assert kwargs["env"]["VLLM_API_KEY"] == "matching-server-key"
        assert kwargs["env"]["VLLM_BASE_URL"] == "https://existing.example/v1"
        assert kwargs["env"]["NGROK_AUTHTOKEN"] == "tunnel-key"
        assert kwargs["env"]["VIRTUAL_ENV"] == str(context["STUDIO_ENV"])
        assert kwargs["start_new_session"] is True
        return child

    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(kaggle_studio, "stop_process", lambda process: stopped.append(process))
    exec(compile(cell_source("supervisor = subprocess.Popen"), "studio-supervisor", "exec"), context)
    assert stopped == [child]
