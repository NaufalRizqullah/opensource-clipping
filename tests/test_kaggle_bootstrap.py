"""Test startup ordering and cleanup without installing ML packages or ngrok."""

import ast
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from types import ModuleType
from urllib.error import HTTPError

import pytest

from scripts import kaggle_studio as bootstrap


def make_environment(tmp_path):
    env_dir = tmp_path / "venv"
    python = bootstrap.environment_python(env_dir)
    python.parent.mkdir(parents=True)
    python.touch()
    (env_dir / "pyvenv.cfg").write_text("include-system-site-packages = false\n", encoding="utf-8")
    return env_dir


def test_runtime_removes_notebook_python_and_pip_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", "notebook-packages")
    monkeypatch.setenv("PYTHONHOME", "notebook-python")
    monkeypatch.setenv("PIP_TARGET", "outside-venv")
    monkeypatch.setenv("NGROK_AUTHTOKEN", "dummy-test-value")
    env = bootstrap.runtime_environment(tmp_path)
    assert "PYTHONPATH" not in env and "PYTHONHOME" not in env and "PIP_TARGET" not in env
    assert env["PYTHON_DOTENV_DISABLED"] == "1"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["KAGGLE_DISABLE_GOOGLE_GENERATIVE_AI_INTEGRATION"] == "1"
    assert env["NGROK_AUTHTOKEN"] == "dummy-test-value"


@pytest.mark.parametrize("backend", [None, "module://matplotlib_inline.backend_inline", "QtAgg"])
def test_managed_renderer_uses_headless_backend_without_changing_kernel(monkeypatch, tmp_path, backend):
    if backend is None:
        monkeypatch.delenv("MPLBACKEND", raising=False)
    else:
        monkeypatch.setenv("MPLBACKEND", backend)
    environment = bootstrap.runtime_environment(tmp_path)
    assert environment["MPLBACKEND"] == "Agg"
    assert bootstrap.os.environ.get("MPLBACKEND") == backend


def test_refuses_environment_with_system_packages(tmp_path):
    env_dir = make_environment(tmp_path)
    (env_dir / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
    with pytest.raises(RuntimeError, match="not isolated"):
        bootstrap.validate_environment(env_dir)


def test_setup_stops_on_install_failure_before_deno_or_pip_check(monkeypatch, tmp_path):
    env_dir = make_environment(tmp_path)
    monkeypatch.setattr(bootstrap, "validate_platform", lambda: None)
    monkeypatch.setattr(bootstrap, "ensure_pip", lambda *a: None)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["check"] is True
        assert "capture_output" not in kwargs
        if "-r" in command:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    monkeypatch.setattr(bootstrap, "install_deno", lambda *a: pytest.fail("Deno must not install after pip failure"))
    with pytest.raises(subprocess.CalledProcessError):
        bootstrap.setup_environment(env_dir, tmp_path)
    assert len(calls) == 2
    assert not any(command[-1] == "check" for command in calls)


def test_readiness_reports_safe_failure_without_opening_tunnel(monkeypatch):
    body = json.dumps({"checks": {"renderer": {
        "ok": False, "error": "ValueError", "message": "private-value-must-not-appear",
        "module": "mediapipe.tasks.python.vision", "code": "import_failed",
        "traceback": [{"module": "matplotlib", "function": "__setitem__", "line": 799}],
    }}, "versions": {"numpy": "1.26.4", "matplotlib": "3.11.2"}}).encode()

    def fail(*args, **kwargs):
        assert kwargs["timeout"] >= 120
        raise HTTPError(args[0], 503, "Unavailable", {}, io.BytesIO(body))

    monkeypatch.setattr(bootstrap, "urlopen", fail)
    with pytest.raises(RuntimeError, match="renderer: ValueError") as error:
        bootstrap.wait_for_readiness(SimpleNamespace(poll=lambda: None), 8000)
    assert "private-value" not in str(error.value)
    assert "import=mediapipe.tasks.python.vision" in str(error.value)
    assert "matplotlib.__setitem__:799" in str(error.value)
    assert "matplotlib=3.11.2" in str(error.value)
    assert "Recreate" not in str(error.value)


@pytest.mark.parametrize("report", [None, [], {"checks": None}, {"checks": {"renderer": []}}])
def test_malformed_readiness_diagnostics_fail_closed(report):
    message = bootstrap.readiness_failure_message(report)
    assert "runtime unavailable" in message
    assert "No public tunnel was opened" in message


@pytest.mark.parametrize("value", [None, [], {}])
def test_malformed_import_metadata_keeps_safe_failure(value):
    message = bootstrap.readiness_failure_message({"checks": {"renderer": {
        "ok": False, "error": value, "module": value, "code": value, "traceback": value,
    }}})
    assert "renderer: unavailable" in message
    assert "No public tunnel was opened" in message


def test_readiness_diagnostics_ignore_raw_text_and_invalid_metadata():
    private = "private-context must not be printed"
    message = bootstrap.readiness_failure_message({
        "checks": {"renderer": {
            "ok": False, "error": private, "message": private, "module": private, "code": private,
            "traceback": [None, {"module": private, "function": private, "line": private}],
            "causes": [{"message": private, "error": "ImportError", "code": "import_failed", "traceback": [
                {"module": "matplotlib.rcsetup", "function": "validate_backend", "line": 287},
            ]}],
        }, private: {"ok": False}},
        "versions": {"numpy": private, private: "1.0"},
    })
    assert private not in message
    assert "renderer: unavailable" in message
    assert "matplotlib.rcsetup.validate_backend:287" in message
    assert "renderer cause: ImportError" in message


def test_readiness_stops_when_child_exits(monkeypatch):
    monkeypatch.setattr(bootstrap, "urlopen", lambda *a, **k: pytest.fail("No request after child exit"))
    with pytest.raises(RuntimeError, match="exit code 1"):
        bootstrap.wait_for_readiness(SimpleNamespace(poll=lambda: 1, returncode=1), 8000)


@pytest.fixture
def server(monkeypatch, tmp_path):
    env_dir = make_environment(tmp_path)
    monkeypatch.setenv("NGROK_AUTHTOKEN", "dummy-test-value")
    monkeypatch.setattr(bootstrap.sys, "executable", str(bootstrap.environment_python(env_dir)))
    monkeypatch.setattr(bootstrap.sys, "prefix", str(env_dir))
    events = []
    child = SimpleNamespace(wait=lambda: 0)

    class Socket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def bind(self, *args): pass

    monkeypatch.setattr(bootstrap.socket, "socket", Socket)

    def launch(command, **kwargs):
        assert command[0] == str(bootstrap.environment_python(env_dir))
        assert command[1:3] == ["-m", "scripts.studio_server"]
        assert kwargs["env"]["PYTHON_DOTENV_DISABLED"] == "1"
        assert kwargs["env"]["MPLBACKEND"] == "Agg"
        events.append("launch")
        return child

    monkeypatch.setattr(bootstrap.subprocess, "Popen", launch)
    monkeypatch.setattr(bootstrap, "wait_for_readiness", lambda *a: events.append("ready"))
    monkeypatch.setattr(bootstrap, "open_tunnel", lambda *a: (
        events.append("tunnel") or SimpleNamespace(public_url="https://example.test"),
        lambda: events.append("close"),
    ))
    monkeypatch.setattr(bootstrap, "stop_process", lambda *a: events.append("stop"))
    return env_dir, tmp_path, events


def test_tunnel_and_banner_only_follow_readiness(server, capsys):
    env_dir, root, events = server
    bootstrap.start_server(env_dir, root)
    assert events == ["launch", "ready", "tunnel", "close", "stop"]
    assert "BACKEND SERVER READY" in capsys.readouterr().out


@pytest.mark.parametrize("exit_code", [130, -bootstrap.signal.SIGINT])
def test_managed_server_interrupt_is_a_normal_shutdown(server, monkeypatch, exit_code):
    env_dir, root, events = server
    monkeypatch.setattr(bootstrap.subprocess, "Popen", lambda *a, **k: SimpleNamespace(wait=lambda: exit_code))
    bootstrap.start_server(env_dir, root)
    assert events[-2:] == ["close", "stop"]


def test_failed_startup_never_opens_tunnel_and_always_stops_child(server, monkeypatch, capsys):
    env_dir, root, events = server
    def fail(*args): raise RuntimeError("renderer unavailable")
    monkeypatch.setattr(bootstrap, "wait_for_readiness", fail)
    with pytest.raises(RuntimeError, match="renderer unavailable"):
        bootstrap.start_server(env_dir, root)
    assert events == ["launch", "stop"]
    assert "BACKEND SERVER READY" not in capsys.readouterr().out


def test_tunnel_cleanup_failure_still_stops_fastapi(server, monkeypatch):
    env_dir, root, events = server
    def close(): raise RuntimeError("simulated ngrok cleanup failure")
    monkeypatch.setattr(bootstrap, "open_tunnel", lambda *a: (SimpleNamespace(public_url="https://example.test"), close))
    with pytest.raises(RuntimeError, match="cleanup failure"):
        bootstrap.start_server(env_dir, root)
    assert events[-1] == "stop"


def test_stuck_process_is_terminated_then_killed():
    calls = []
    def wait(timeout):
        calls.append(("wait", timeout))
        if len([c for c in calls if isinstance(c, tuple)]) < 3:
            raise subprocess.TimeoutExpired("server", timeout)
    process = SimpleNamespace(
        poll=lambda: None, send_signal=lambda signal: calls.append("interrupt"),
        wait=wait, terminate=lambda: calls.append("terminate"), kill=lambda: calls.append("kill"),
    )
    bootstrap.stop_process(process)
    assert calls == ["interrupt", ("wait", 20), "terminate", ("wait", 10), "kill", ("wait", 10)]


def test_notebook_uses_supervisor_and_contains_no_destructive_setup():
    notebook = json.loads((bootstrap.PROJECT_ROOT / "notebooks/kaggle-studio-server.ipynb").read_text(encoding="utf-8"))
    source = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            code = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
            ast.parse(code)
            assert cell.get("outputs", []) == []
            source.append(code)
    combined = "\n".join(source)
    assert "rm -rf" not in combined and "%%capture" not in combined
    assert "nest_asyncio" not in combined and "server.serve" not in combined
    assert "stop_process(supervisor)" in combined
    assert "runtime_environment(STUDIO_ENV)" in combined


def test_start_rejects_system_python_even_if_executable_matches(monkeypatch, tmp_path):
    env_dir = make_environment(tmp_path)
    monkeypatch.setenv("NGROK_AUTHTOKEN", "dummy-test-value")
    monkeypatch.setattr(bootstrap.sys, "executable", str(bootstrap.environment_python(env_dir)))
    monkeypatch.setattr(bootstrap.sys, "prefix", str(tmp_path / "system-python"))
    monkeypatch.setattr(bootstrap.subprocess, "Popen", lambda *a, **k: pytest.fail("No launch outside venv"))
    with pytest.raises(RuntimeError, match="isolated environment's Python"):
        bootstrap.start_server(env_dir, tmp_path)


def test_new_environment_does_not_call_ensurepip(monkeypatch, tmp_path):
    env_dir = tmp_path / "venv"
    calls = []
    monkeypatch.setattr(bootstrap, "validate_platform", lambda: None)
    def builder(**options):
        assert options == {"with_pip": False, "system_site_packages": False}
        return SimpleNamespace(create=lambda path: (calls.append("create"), make_environment(tmp_path)))
    monkeypatch.setattr(bootstrap.venv, "EnvBuilder", builder)
    monkeypatch.setattr(bootstrap, "ensure_pip", lambda *a: calls.append("pip"))
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **k: calls.append("install"))
    monkeypatch.setattr(bootstrap, "install_deno", lambda *a: calls.append("deno"))
    bootstrap.setup_environment(env_dir, tmp_path)
    assert calls == ["create", "pip", "install", "install", "install", "deno"]


def test_partial_environment_is_repaired_in_place_and_failure_stops_setup(monkeypatch, tmp_path):
    env_dir = make_environment(tmp_path)
    marker = env_dir / "preserved.txt"
    marker.write_text("keep")
    monkeypatch.setattr(bootstrap, "validate_platform", lambda: None)
    monkeypatch.setattr(bootstrap.venv, "EnvBuilder", lambda **k: pytest.fail("Do not recreate existing venv"))
    def fail(*a): raise RuntimeError("pip bootstrap failed")
    monkeypatch.setattr(bootstrap, "ensure_pip", fail)
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **k: pytest.fail("No dependency install"))
    monkeypatch.setattr(bootstrap, "install_deno", lambda *a: pytest.fail("No Deno install"))
    with pytest.raises(RuntimeError, match="pip bootstrap failed"):
        bootstrap.setup_environment(env_dir, tmp_path)
    assert marker.read_text() == "keep"


@pytest.mark.parametrize("host_available", [True, False])
def test_missing_pip_uses_host_manager_or_official_target_bootstrap(monkeypatch, tmp_path, host_available):
    env_dir = make_environment(tmp_path)
    target = str(bootstrap.environment_python(env_dir))
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if "-c" in command:
            return SimpleNamespace(stdout=json.dumps([str(env_dir), str(tmp_path / "base")]), returncode=0)
        if command[-1] == "--version":
            return SimpleNamespace(returncode=1 if len(calls) == 2 else 0)
        if "--python" in command:
            assert command[command.index("--python") + 1] == target
            if not host_available:
                raise subprocess.CalledProcessError(1, command)
        else:
            assert command[:2] == [target, "-I"]
            assert Path(command[2]).name == "get-pip.py"
            assert Path(command[2]).read_bytes() == b"# official bootstrap fixture"
        assert kwargs["env"]["PIP_REQUIRE_VIRTUALENV"] == "true"
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    def download(url, **kwargs):
        assert not host_available
        assert url == "https://bootstrap.pypa.io/get-pip.py"
        return io.BytesIO(b"# official bootstrap fixture")
    monkeypatch.setattr(bootstrap, "urlopen", download)
    bootstrap.ensure_pip(env_dir, tmp_path)
    assert len(calls) == (4 if host_available else 5)
    assert calls[-1] == [target, "-I", "-m", "pip", "--version"]


def test_existing_pip_skips_bootstrap(monkeypatch, tmp_path):
    env_dir = make_environment(tmp_path)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert "install" not in command
        return SimpleNamespace(stdout=json.dumps([str(env_dir), str(tmp_path / "base")]), returncode=0)
    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    monkeypatch.setattr(bootstrap, "urlopen", lambda *a, **k: pytest.fail("No download"))
    bootstrap.ensure_pip(env_dir, tmp_path)
    assert len(calls) == 2


def test_target_interpreter_mismatch_is_rejected_before_any_install(monkeypatch, tmp_path):
    env_dir = make_environment(tmp_path)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps([str(tmp_path), str(tmp_path)]), returncode=0)
    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="not using the isolated environment"):
        bootstrap.ensure_pip(env_dir, tmp_path)
    assert len(calls) == 1


def _secret_cell(monkeypatch, reader):
    """Run only the repository template with fake Secrets and an empty environment."""
    monkeypatch.setattr(bootstrap.os, "environ", {})
    monkeypatch.setitem(sys.modules, "kaggle_secrets", SimpleNamespace(
        UserSecretsClient=lambda: SimpleNamespace(get_secret=reader),
    ))
    google = ModuleType("google")
    google.__path__ = []
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(userdata=SimpleNamespace(get=lambda name: "")))
    notebook = json.loads((bootstrap.PROJECT_ROOT / "notebooks/kaggle-studio-server.ipynb").read_text(encoding="utf-8"))
    source = notebook["cells"][6]["source"]
    exec(compile(source, "notebook-secret-cell", "exec"), {})


def test_notebook_exports_attached_secret_without_printing_it(monkeypatch, capsys, tmp_path):
    dummy = "dummy-not-a-real-token"
    _secret_cell(monkeypatch, lambda name: f"  {dummy}\n" if name == "NGROK_AUTHTOKEN" else "")
    assert bootstrap.runtime_environment(tmp_path)["NGROK_AUTHTOKEN"] == dummy
    output = capsys.readouterr().out
    assert "NGROK_AUTHTOKEN: configured" in output
    assert dummy not in output


def test_notebook_stops_with_safe_guidance_when_secret_reader_fails(monkeypatch, capsys):
    def unavailable(name):
        raise RuntimeError("private-provider-response-must-not-be-printed")
    with pytest.raises(RuntimeError, match="Attach this exact secret label"):
        _secret_cell(monkeypatch, unavailable)
    output = capsys.readouterr().out
    assert "NGROK_AUTHTOKEN could not be read (RuntimeError)" in output
    assert "private-provider-response" not in output


@pytest.mark.parametrize("value", ["", "   "])
def test_missing_or_blank_ngrok_token_stops_before_launch(monkeypatch, tmp_path, value):
    env_dir = make_environment(tmp_path)
    monkeypatch.setenv("NGROK_AUTHTOKEN", value)
    monkeypatch.setattr(bootstrap.subprocess, "Popen", lambda *a, **k: pytest.fail("Must not launch"))
    with pytest.raises(RuntimeError, match="Attach the secret with this exact label"):
        bootstrap.start_server(env_dir)
