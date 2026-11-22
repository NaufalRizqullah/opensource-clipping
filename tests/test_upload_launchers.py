"""Exercise relocated upload CLIs without SDKs, credentials, or network calls."""

import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script,package", [
    ("run_upload.py", "youtube_uploader"),
    ("run_fb_upload.py", "facebook_uploader"),
])
def test_direct_execution_from_another_directory_without_sdks(tmp_path, script, package):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": "",
           "PYTHON_DOTENV_DISABLED": "1"}
    script_path = ROOT / "scripts" / script
    help_result = subprocess.run(
        [sys.executable, "-S", str(script_path), "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
        encoding="utf-8", timeout=15,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--manifest-file" in help_result.stdout
    assert "--test-mode" in help_result.stdout

    # Verify the repository packages are discoverable after the script bootstrap,
    # without importing SDKs or starting an upload.
    import_result = subprocess.run(
        [sys.executable, "-S", "-c",
         "import importlib.util, pathlib, runpy, sys; "
         "runpy.run_path(sys.argv[1]); "
         "spec = importlib.util.find_spec(sys.argv[2]); "
         "assert spec is not None; "
         "assert pathlib.Path(spec.origin) == "
         "pathlib.Path(sys.argv[1]).resolve().parents[1] / sys.argv[2] / '__init__.py'",
         str(script_path), package],
        cwd=tmp_path, env=env, capture_output=True, text=True,
        encoding="utf-8", timeout=15,
    )
    assert import_result.returncode == 0, import_result.stderr
    assert not list(tmp_path.iterdir())


@pytest.fixture
def launchers(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    youtube_upload = Mock()
    facebook_upload = Mock()
    monkeypatch.setitem(sys.modules, "youtube_uploader", SimpleNamespace(
        upload_manifest_to_youtube=youtube_upload,
    ))
    monkeypatch.setitem(sys.modules, "facebook_uploader", SimpleNamespace(
        upload_manifest_to_facebook=facebook_upload,
    ))
    monkeypatch.setitem(sys.modules, "dotenv", SimpleNamespace(load_dotenv=Mock()))
    return SimpleNamespace(
        youtube=importlib.import_module("scripts.run_upload"),
        facebook=importlib.import_module("scripts.run_fb_upload"),
        youtube_upload=youtube_upload,
        facebook_upload=facebook_upload,
    )


@pytest.mark.parametrize("custom", [False, True])
def test_youtube_forwards_supported_arguments(launchers, monkeypatch, custom):
    token = Path("custom credentials/token.json" if custom else ".credentials/youtube_token.json")
    token.parent.mkdir()
    token.write_text("{}", encoding="utf-8")
    argv = ["scripts/run_upload.py"]
    expected = dict(
        token_file=str(token) if custom else ".credentials/youtube_token.json",
        manifest_file="outputs/render_manifest.json",
        result_file="outputs/youtube_upload_results.json",
        updated_manifest_file="outputs/render_manifest_uploaded.json",
        tz_name="Asia/Makassar", interval_hours=24, start_local=None, test_mode=False,
    )
    if custom:
        expected.update(
            manifest_file="input clips/manifest.json", result_file="results/youtube.json",
            updated_manifest_file="results/manifest.json", tz_name="Asia/Jakarta",
            interval_hours=48, start_local="2026-12-01 10:00", test_mode=True,
        )
        argv.extend([
            "--token-file", str(token), "--manifest-file", expected["manifest_file"],
            "--result-file", expected["result_file"],
            "--updated-manifest", expected["updated_manifest_file"],
            "--tz-name", expected["tz_name"], "--interval-hours", "48",
            "--start-local", expected["start_local"], "--test-mode",
        ])
    monkeypatch.setattr(sys, "argv", argv)
    launchers.youtube.main()
    launchers.youtube_upload.assert_called_once_with(**expected)


@pytest.mark.parametrize("custom", [False, True])
def test_facebook_forwards_arguments_and_uses_environment_timezone(launchers, monkeypatch, custom):
    monkeypatch.setenv("META_PAGE_ACCESS_TOKEN", "synthetic-test-token")
    monkeypatch.setenv("META_PAGE_ID", "synthetic-test-page")
    monkeypatch.setenv("APP_TIMEZONE", "Asia/Jakarta")
    argv = ["scripts/run_fb_upload.py"]
    expected = dict(
        manifest_file="outputs/render_manifest.json", result_file="outputs/fb_upload_results.json",
        updated_manifest_file="outputs/render_manifest_fb_uploaded.json",
        tz_name="Asia/Jakarta", interval_hours=5, test_mode=False,
    )
    if custom:
        expected.update(
            manifest_file="input clips/manifest.json", result_file="results/facebook.json",
            updated_manifest_file="results/manifest.json", tz_name="UTC",
            interval_hours=3, test_mode=True,
        )
        argv.extend([
            "--manifest-file", expected["manifest_file"], "--result-file", expected["result_file"],
            "--updated-manifest", expected["updated_manifest_file"],
            "--tz-name", "UTC", "--interval-hours", "3", "--test-mode",
        ])
    monkeypatch.setattr(sys, "argv", argv)
    launchers.facebook.main()
    launchers.facebook_upload.assert_called_once_with(**expected)


@pytest.mark.parametrize("flags", [
    ["--safety-config", "unused-safety.json"],
    ["--safety-config", ""],
    ["--no-approval"],
])
def test_retired_safety_options_stop_before_upload(launchers, monkeypatch, capsys, tmp_path, flags):
    monkeypatch.setattr(sys, "argv", ["scripts/run_upload.py", *flags])
    with pytest.raises(SystemExit) as exc:
        launchers.youtube.main()
    assert exc.value.code == 2
    assert "no longer supported" in capsys.readouterr().err
    launchers.youtube_upload.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_youtube_missing_credentials_stop_before_upload(launchers, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["scripts/run_upload.py"])
    with pytest.raises(SystemExit) as exc:
        launchers.youtube.main()
    assert exc.value.code == 1
    launchers.youtube_upload.assert_not_called()


@pytest.mark.parametrize("missing", ["META_PAGE_ACCESS_TOKEN", "META_PAGE_ID"])
def test_facebook_missing_credentials_stop_before_upload(launchers, monkeypatch, missing):
    monkeypatch.setenv("META_PAGE_ACCESS_TOKEN", "synthetic-test-token")
    monkeypatch.setenv("META_PAGE_ID", "synthetic-test-page")
    monkeypatch.delenv(missing)
    monkeypatch.setattr(sys, "argv", ["scripts/run_fb_upload.py"])
    with pytest.raises(SystemExit) as exc:
        launchers.facebook.main()
    assert exc.value.code == 1
    launchers.facebook_upload.assert_not_called()
