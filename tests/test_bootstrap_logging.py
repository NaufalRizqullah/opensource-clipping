"""Verify saved startup evidence without installing dependencies or opening tunnels."""

import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from scripts import kaggle_studio as bootstrap
from scripts import studio_logging


@pytest.fixture
def log_dir(monkeypatch, tmp_path):
    studio_logging.close_logging()
    monkeypatch.setenv("OSC_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("OSC_LOG_RUN_ID", raising=False)
    yield tmp_path / "logs"
    studio_logging.close_logging()


def read_events(log_dir):
    return [json.loads(line) for line in (log_dir / "kaggle-studio.jsonl").read_text().splitlines()]


def test_installer_failure_saved_with_stage_exit_code_and_traceback(monkeypatch, log_dir, capsys):
    private = "synthetic-password-must-not-be-saved"

    def install_failure(*args):
        with bootstrap.logged_stage("dependencies"):
            raise subprocess.CalledProcessError(7, ["pip", "--password", private], output=private, stderr=private)

    monkeypatch.setattr(bootstrap, "setup_environment", install_failure)
    assert bootstrap.main(["setup"]) == 1
    events = read_events(log_dir)
    failed = next(event for event in events if event["event"] == "stage.failed")
    assert failed["stage"] == "dependencies"
    assert failed["exit_code"] == 7
    assert failed["error"]["type"] == "CalledProcessError"
    assert failed["error"]["traceback"]
    assert events[-1]["event"] == "session.failed"
    assert len({event["run_id"] for event in events}) == 1
    assert private not in json.dumps(events)
    console = capsys.readouterr()
    assert private not in console.out + console.err


def test_failed_readiness_saved_before_supervisor_raises(monkeypatch, log_dir):
    studio_logging.configure_logging("bootstrap")
    body = json.dumps({"ready": False, "checks": {"renderer": {
        "ok": False, "error": "ValueError", "module": "mediapipe.tasks.python.vision",
        "code": "import_failed", "message": "synthetic-secret-not-for-file",
        "traceback": [{"module": "matplotlib", "function": "__setitem__", "line": 784}],
    }}, "versions": {"matplotlib": "3.11.2"}}).encode()

    def fail(*args, **kwargs):
        raise HTTPError(args[0], 503, "Unavailable", {}, io.BytesIO(body))

    monkeypatch.setattr(bootstrap, "urlopen", fail)
    with pytest.raises(RuntimeError, match="renderer: ValueError"):
        bootstrap.wait_for_readiness(SimpleNamespace(poll=lambda: None), 8000)
    events = read_events(log_dir)
    event = next(event for event in events if event["event"] == "readiness.failed")
    renderer = event["diagnostics"]["checks"]["renderer"]
    assert renderer["module"] == "mediapipe.tasks.python.vision"
    assert renderer["traceback"][0]["line"] == 784
    assert "synthetic-secret" not in json.dumps(events)


def test_standalone_bootstrap_imports_logger_without_site_packages():
    result = subprocess.run(
        [sys.executable, "-E", "-S", str(Path(bootstrap.__file__)), "--help"],
        cwd=bootstrap.PROJECT_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--venv" in result.stdout
