"""Notebook provider wiring with fake secrets/processes; no installs or inference."""

import json
import os
from pathlib import Path
import secrets
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import kaggle_python, kaggle_storage, kaggle_studio


ROOT = Path(__file__).resolve().parents[1]
CONFIG = "# AI provider configuration"
CREDENTIALS = "# Load provider credentials"
PREPARE = "# Prepare isolated application"
START = "# Start local vLLM"
STOP = "# Stop the managed local server"
RUN = "# Run the clipping pipeline"


def cell_source(notebook, marker):
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    sources = ["".join(cell["source"]) for cell in cells
               if cell["cell_type"] == "code" and marker in "".join(cell["source"])]
    assert len(sources) == 1
    return sources[0]


def execute(notebook, marker, scope, replacements=()):
    source = cell_source(notebook, marker)
    for old, new in replacements:
        assert old in source
        source = source.replace(old, new, 1)
    exec(compile(source, str(notebook), "exec"), scope)


def configure(notebook, provider="gemini", local=False):
    scope = {}
    execute(notebook, CONFIG, scope, [
        ('AI_PROVIDER = "gemini"', f"AI_PROVIDER = {provider!r}"),
        ("START_LOCAL_VLLM = False", f"START_LOCAL_VLLM = {local!r}"),
    ])
    return scope


@pytest.fixture(params=["Quick_Start.ipynb", "Lib_OpenSource_Clipping.ipynb"])
def notebook(request, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(os, "environ", {"PYTHON_DOTENV_DISABLED": "1"})
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, "google.colab", None)
    monkeypatch.setitem(sys.modules, "kaggle_secrets", None)

    def forbidden(*args, **kwargs):
        pytest.fail("Notebook tests must not start processes, install packages, or download models")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(kaggle_python, "ensure_python", forbidden)
    monkeypatch.setattr(kaggle_storage, "prepare_storage", forbidden)
    return ROOT / "notebooks" / request.param


def test_default_gemini_skips_local_server_and_keeps_env_file(notebook, monkeypatch, capsys):
    monkeypatch.setenv("GOOGLE_API_KEY", "synthetic-google-key")
    scope = configure(notebook)
    execute(notebook, CREDENTIALS, scope)
    execute(notebook, PREPARE, scope)
    execute(notebook, START, scope)
    execute(notebook, STOP, scope)
    assert scope["AI_PROVIDER"] == "gemini"
    assert scope["APP_PYTHON"] == sys.executable and scope["APP_ENV"] is None
    assert "GOOGLE_API_KEY=synthetic-google-key" in Path(".env").read_text()
    assert "synthetic-google-key" not in capsys.readouterr().out


def test_gemini_requires_its_key(notebook):
    scope = configure(notebook)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        execute(notebook, CREDENTIALS, scope)


@pytest.mark.parametrize("provider,local", [("unsupported", False), ("gemini", True)])
def test_bad_provider_choice_fails_before_installation(notebook, provider, local):
    with pytest.raises(ValueError):
        configure(notebook, provider, local)


@pytest.mark.parametrize("reader", ["kaggle", "colab"])
def test_vllm_secret_reader_does_not_request_google_key(notebook, monkeypatch, capsys, reader):
    requested = []

    def get_secret(name):
        requested.append(name)
        if name == "VLLM_API_KEY":
            return "synthetic-server-key"
        raise RuntimeError("synthetic missing optional secret")

    if reader == "kaggle":
        monkeypatch.setitem(sys.modules, "kaggle_secrets", SimpleNamespace(
            UserSecretsClient=lambda: SimpleNamespace(get_secret=get_secret)))
    else:
        monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(userdata=SimpleNamespace(get=get_secret)))
    scope = configure(notebook, "vllm")
    execute(notebook, CREDENTIALS, scope)
    assert os.environ["VLLM_API_KEY"] == "synthetic-server-key"
    assert "GOOGLE_API_KEY" not in requested
    assert "synthetic-server-key" not in capsys.readouterr().out
    assert not Path(".env").exists()


def test_external_vllm_settings_and_key_are_preserved(notebook, monkeypatch, capsys):
    monkeypatch.setenv("VLLM_BASE_URL", "https://private-server.invalid/inference/v1")
    monkeypatch.setenv("VLLM_MODEL", "served-model")
    monkeypatch.setenv("VLLM_API_KEY", "existing-server-key")
    monkeypatch.setenv("VLLM_CONTEXT_LIMIT", "8192")
    monkeypatch.setenv("VLLM_MAX_TOKENS", "1024")
    monkeypatch.setenv("VLLM_TIMEOUT", "120.5")
    monkeypatch.setattr(secrets, "token_urlsafe", lambda *_: pytest.fail("Do not replace the external key"))
    scope = configure(notebook, "vllm")
    execute(notebook, CREDENTIALS, scope)
    execute(notebook, PREPARE, scope)
    execute(notebook, START, scope)
    execute(notebook, STOP, scope, [("STOP_LOCAL_VLLM = False", "STOP_LOCAL_VLLM = True")])
    assert os.environ["VLLM_BASE_URL"] == "https://private-server.invalid/inference"
    assert os.environ["VLLM_MODEL"] == "served-model"
    assert os.environ["VLLM_API_KEY"] == "existing-server-key"
    assert os.environ["VLLM_CONTEXT_LIMIT"] == "8192"
    assert os.environ["VLLM_MAX_TOKENS"] == "1024"
    assert os.environ["VLLM_TIMEOUT"] == "120.5"
    assert "existing-server-key" not in capsys.readouterr().out


def test_local_key_is_generated_once_and_not_persisted(notebook, monkeypatch, capsys):
    generated = []
    monkeypatch.setattr(secrets, "token_urlsafe", lambda n: generated.append(n) or "synthetic-local-key")
    scope = configure(notebook, "vllm", True)
    execute(notebook, CREDENTIALS, scope)
    execute(notebook, CREDENTIALS, scope)
    assert generated == [32]
    assert os.environ["VLLM_BASE_URL"] == "http://127.0.0.1:8001"
    assert "synthetic-local-key" not in capsys.readouterr().out
    assert not Path(".env").exists()


@pytest.mark.parametrize("setting,value", [("VLLM_BASE_URL", "file:///invalid"), ("VLLM_MAX_TOKENS", "16384")])
def test_invalid_vllm_configuration_fails_before_setup(notebook, setting, value):
    scope = configure(notebook, "vllm")
    scope[setting] = value
    with pytest.raises(ValueError):
        execute(notebook, CREDENTIALS, scope)


@pytest.mark.parametrize("local", [False, True])
def test_dependency_installation_uses_selected_mode(notebook, monkeypatch, local):
    scope = configure(notebook, "vllm" if local else "gemini", local)
    source = cell_source(notebook, '"-m", "pip", "install"')
    # System shell commands are outside this test; execute the Python pip branch only.
    source = "\n".join(line for line in source.splitlines() if not line.startswith(("!", "%")))
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: calls.append((argv, kw)))
    os.environ["PATH"] = "synthetic-path"
    exec(compile(source, str(notebook), "exec"), scope)
    assert len(calls) == (0 if local else 1)
    if calls:
        assert calls[0][0] == [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        assert calls[0][1]["check"] is True


@pytest.fixture
def local_context(notebook, monkeypatch, tmp_path):
    scope = configure(notebook, "vllm", True)
    execute(notebook, CREDENTIALS, scope)
    root = tmp_path / "scratch"
    storage = SimpleNamespace(root=root, python_cache=root / "python", vllm_env=root / "vllm",
                              cache_environment=lambda: {"HF_HUB_CACHE": str(root / "models")})
    monkeypatch.setattr(kaggle_storage, "prepare_storage", lambda *args, **kw: storage)
    monkeypatch.setattr(kaggle_storage, "check_storage", lambda *args: None)
    monkeypatch.setattr(kaggle_python, "ensure_python", lambda *_: Path("/managed/python3.12"))
    return scope


def test_local_lifecycle_uses_separate_environments(notebook, local_context, monkeypatch):
    calls = []

    def run(argv, **kwargs):
        assert kwargs["check"] is True
        assert kwargs["cwd"] == local_context["PROJECT_DIR"]
        calls.append((argv, kwargs))

    monkeypatch.setattr(subprocess, "run", run)
    execute(notebook, PREPARE, local_context)
    execute(notebook, START, local_context)
    assert [argv[2] for argv, _ in calls[:4]] == ["preflight", "setup", "setup", "start"]
    assert calls[0][0][0] == sys.executable
    assert all(argv[0] == str(Path("/managed/python3.12")) for argv, _ in calls[1:4])
    app_setup, server_setup = calls[1][0], calls[2][0]
    assert app_setup[app_setup.index("--torch-profile") + 1] == "cpu"
    assert app_setup[app_setup.index("--venv") + 1] != server_setup[server_setup.index("--venv") + 1]
    eval_argv, eval_kwargs = calls[-1]
    assert eval_argv[:2] == [local_context["APP_PYTHON"], "scripts/eval_local_llm.py"]
    assert eval_kwargs["env"]["VIRTUAL_ENV"] == str(local_context["APP_ENV"])
    assert eval_kwargs["env"]["VLLM_API_KEY"] == os.environ["VLLM_API_KEY"]
    assert local_context["LOCAL_VLLM_READY"] is True
    execute(notebook, STOP, local_context)
    assert len(calls) == 5
    execute(notebook, STOP, local_context, [("STOP_LOCAL_VLLM = False", "STOP_LOCAL_VLLM = True")])
    assert calls[-1][0][1:] == ["scripts/kaggle_vllm.py", "stop"]
    assert local_context["LOCAL_VLLM_READY"] is False


@pytest.mark.parametrize("failure_step", ["preflight", "setup", "start", "eval"])
def test_local_failure_blocks_pipeline(notebook, local_context, monkeypatch, failure_step):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        step = "eval" if argv[1] == "scripts/eval_local_llm.py" else argv[2]
        if step == failure_step:
            raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        execute(notebook, PREPARE, local_context)
        execute(notebook, START, local_context)
    assert not local_context["LOCAL_VLLM_READY"]
    with pytest.raises(RuntimeError, match="Start/Eval"):
        execute(notebook, RUN, local_context)
    if failure_step == "preflight":
        assert len(calls) == 1


@pytest.mark.parametrize("provider,local", [("gemini", False), ("vllm", False), ("vllm", True)])
def test_pipeline_passes_provider_cookies_and_correct_whisper_mode(notebook, monkeypatch, provider, local):
    from clipping.config import build_config

    monkeypatch.setenv("GOOGLE_API_KEY", "synthetic-google-key")
    scope = configure(notebook, provider, local)
    execute(notebook, CREDENTIALS, scope)
    cookie_path = str(Path.cwd() / "cookie folder" / "cookies.txt")
    scope.update(
        LOCAL_VLLM_READY=True, YOUTUBE_URL="https://youtube.com/watch?v=test",
        TOTAL_CLIPS=3, ASPECT_RATIO="9:16", FONT_STYLE="DEFAULT", HOOK_DURATION=3,
        WORDS_PER_SUBTITLE=5, WHISPER_MODEL="large-v3", WHISPER_DEVICE="cuda",
        WHISPER_COMPUTE_TYPE="float16", YT_COOKIES_ARG="--yt-cookies " + shlex.quote(cookie_path),
    )
    if local:
        scope.update(APP_ENV=Path("/isolated/app"), APP_PYTHON="/isolated/app/bin/python",
                     runtime_environment=kaggle_studio.runtime_environment)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kwargs: calls.append((argv, kwargs)))
    execute(notebook, RUN, scope)
    assert len(calls) == 1
    argv, kwargs = calls[0]
    cfg = build_config(argv[2:])
    assert cfg.ai_provider == provider
    assert cfg.yt_cookies == cookie_path
    assert cfg.whisper_device == ("cpu" if local else "cuda")
    assert cfg.whisper_compute_type == ("int8" if local else "float16")
    assert cfg.vllm_chunk_transcript is (provider == "vllm")
    assert argv[0] == scope["APP_PYTHON"]
    assert kwargs["check"] is True
    assert "synthetic-google-key" not in argv
    if local:
        assert kwargs["env"]["VIRTUAL_ENV"] == str(scope["APP_ENV"])
