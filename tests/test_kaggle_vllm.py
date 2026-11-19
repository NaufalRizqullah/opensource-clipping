"""Offline regression coverage for the opt-in Kaggle server helper.

All process, GPU, installer and HTTP operations are replaced with fakes. Run
these tests in the Kaggle application environment, not on the user's PC.
"""

import io
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from scripts import kaggle_vllm as helper


def arguments(tmp_path, **overrides):
    values = dict(venv=tmp_path / "server-venv", state=tmp_path / "server.json",
                  model=helper.DEFAULT_MODEL, revision=None, port=8001, context=16384,
                  max_num_seqs=1, gpu_memory_utilization=0.85, gpu_index=0,
                  startup_timeout=60)
    values.update(overrides)
    return SimpleNamespace(**values)


def saved_state(**overrides):
    values = dict(pid=12001, identity=["123456", "boot-id"], owner="test-owner",
                  port=8001, model=helper.DEFAULT_MODEL, log="server.log")
    values.update(overrides)
    return values


def test_preset_pins_qat_checkpoint_and_constrains_single_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "secret-not-in-command")
    command = helper.build_server_command(arguments(tmp_path))
    for flag, expected in {"--host": "127.0.0.1", "--dtype": "half",
                           "--tensor-parallel-size": "1", "--max-model-len": "16384",
                           "--max-num-seqs": "1", "--port": "8001",
                           "--revision": helper.DEFAULT_REVISION,
                           "--tokenizer-revision": helper.DEFAULT_REVISION}.items():
        assert command[command.index(flag) + 1] == expected
    assert helper.DEFAULT_MODEL in command
    assert "--enforce-eager" in command and "--language-model-only" in command
    assert json.loads(command[command.index("--default-chat-template-kwargs") + 1]) == {"enable_thinking": False}
    assert "--quantization" not in command  # Read the checkpoint's QAT configuration.
    assert "secret-not-in-command" not in " ".join(command)
    assert "--no-enable-log-requests" in command and "--no-enable-log-outputs" in command


def test_custom_checkpoint_does_not_inherit_an_unrelated_revision(tmp_path):
    command = helper.build_server_command(arguments(tmp_path, model="owner/gemma4-custom"))
    assert "--revision" not in command
    command = helper.build_server_command(arguments(tmp_path, model="owner/gemma4-custom", revision="custom-commit"))
    assert command[command.index("--revision") + 1] == "custom-commit"


@pytest.mark.parametrize("capability", [7.5, 8.0, 8.6])
def test_attention_compat_runs_only_in_turing_server_environment(monkeypatch, tmp_path, capability):
    calls = []
    environment = {"CUDA_VISIBLE_DEVICES": "GPU-selected"}

    def run(command, **kwargs):
        calls.append(command)
        assert command == [str(helper.environment_python(tmp_path)),
                           str(helper.ROOT / "scripts/kaggle_vllm_compat.py"),
                           "--venv", str(tmp_path)]
        assert kwargs["env"] is environment
        assert kwargs["check"] is True and kwargs["timeout"] == 30

    monkeypatch.setattr(helper.subprocess, "run", run)
    helper.prepare_attention_compat(tmp_path, {"compute_capability": capability}, environment)
    assert len(calls) == (1 if capability == 7.5 else 0)


@pytest.mark.parametrize("name,capability,total,expected_error", [
    ("Tesla T4", "7.5", "15360", None),
    ("Tesla P100", "6.0", "16384", "SM 7.5"),
    ("Small GPU", "8.6", "8192", "16 GB-class"),
])
def test_gpu_preflight_checks_architecture_and_vram(monkeypatch, name, capability, total, expected_error):
    monkeypatch.setattr(helper, "require_platform", lambda: None)

    def run(command, **kwargs):
        assert command[0] == "nvidia-smi" and kwargs["check"] is True
        return SimpleNamespace(stdout=f"0, GPU-test, {name}, {total}, {total}, {capability}, 580.00\n")

    monkeypatch.setattr(helper.subprocess, "run", run)
    if expected_error:
        with pytest.raises(RuntimeError, match=expected_error):
            helper.gpu_preflight()
    else:
        assert helper.gpu_preflight()["uuid"] == "GPU-test"


def test_cpu_session_stops_without_installing(monkeypatch):
    monkeypatch.setattr(helper, "require_platform", lambda: None)
    monkeypatch.setattr(helper.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=""))
    with pytest.raises(RuntimeError, match="was not found"):
        helper.gpu_preflight()


@pytest.mark.parametrize("timeout", ["nan", "inf", "0", "-1"])
def test_invalid_timeout_is_rejected_before_platform_or_process_access(monkeypatch, timeout):
    monkeypatch.setattr(helper, "require_platform", lambda: pytest.fail("No platform or process access"))
    with pytest.raises(SystemExit) as caught:
        helper.main(["start", "--startup-timeout", timeout])
    assert caught.value.code == 2


def test_setup_refuses_studio_or_active_python_environment(monkeypatch):
    monkeypatch.setattr(helper, "require_platform", lambda: None)
    monkeypatch.setattr(helper.subprocess, "run", lambda *a, **kw: pytest.fail("No installer may run"))
    for directory in (helper.ROOT / ".cache/kaggle-studio/venv", Path(helper.sys.prefix)):
        with pytest.raises(RuntimeError, match="separate vLLM environment"):
            helper.setup(directory, helper.DEFAULT_VERSION)


def test_setup_preserves_pinned_dependency_pair(monkeypatch, tmp_path):
    monkeypatch.setattr(helper, "require_platform", lambda: None)
    monkeypatch.setattr(helper.subprocess, "run", lambda *a, **kw: pytest.fail("No installer may run"))
    with pytest.raises(RuntimeError, match="pins vLLM and Transformers"):
        helper.setup(tmp_path / "server", "999.0.0")


def test_setup_uses_only_target_python_and_stops_on_pip_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(helper, "require_platform", lambda: None)
    env_dir = tmp_path / "server"
    env_dir.mkdir()
    monkeypatch.setattr(helper, "validate_environment", lambda directory: None)
    monkeypatch.setattr(helper, "ensure_pip", lambda *a: None)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[0] == str(helper.environment_python(env_dir))
        assert kwargs["check"] is True
        if "install" in command:
            requirements = Path(command[command.index("-r") + 1]).read_text()
            assert "vllm-0.32.0%2Bcu129" in requirements
            assert "transformers==5.16.1" in requirements
            raise helper.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(helper.subprocess, "run", run)
    with pytest.raises(helper.subprocess.CalledProcessError):
        helper.setup(env_dir, helper.DEFAULT_VERSION)
    assert len(calls) == 1


def test_invalid_state_refuses_to_guess_process_ownership(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"pid": 12001}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Invalid vLLM state"):
        helper.read_state(path)


def test_reused_pid_is_never_signalled(monkeypatch, tmp_path):
    monkeypatch.setattr(helper, "process_identity", lambda pid: ("different-start", "boot-id"))
    monkeypatch.setattr(helper.os, "killpg", lambda *a: pytest.fail("Unrelated process must not be signalled"), raising=False)
    with pytest.raises(RuntimeError, match="another process"):
        helper.stop_owned(tmp_path / "state.json", saved_state())


def test_missing_owner_marker_is_never_adopted(monkeypatch):
    monkeypatch.setattr(helper, "process_identity", lambda pid: ("123456", "boot-id"))
    monkeypatch.setattr(Path, "read_bytes", lambda path: b"OTHER=value\0")
    monkeypatch.setattr(helper.os, "getpgid", lambda pid: pid, raising=False)
    with pytest.raises(RuntimeError, match="ownership could not be verified"):
        helper.is_owned(saved_state())


def test_dead_process_state_is_removed_without_signalling(monkeypatch, tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps(saved_state()), encoding="utf-8")
    monkeypatch.setattr(helper, "process_identity", lambda pid: None)
    monkeypatch.setattr(helper.os, "killpg", lambda *a: pytest.fail("Dead process must not be signalled"), raising=False)
    helper.stop_owned(path, saved_state())
    assert not path.exists()


def test_readiness_requires_exact_model_and_preserves_auth(monkeypatch):
    monkeypatch.setattr(helper, "is_owned", lambda state: True)
    models = iter([{"data": [{"id": "other-model"}]}, {"data": [{"id": helper.DEFAULT_MODEL}]}])
    calls = []

    def response(url):
        calls.append(url)
        return next(models)

    monkeypatch.setattr(helper, "request_json", response)
    monkeypatch.setattr(helper.time, "sleep", lambda seconds: None)
    helper.wait_ready(saved_state(), 60)
    assert calls == ["http://127.0.0.1:8001/v1/models"] * 2


def test_auth_error_fails_without_retries_or_fallback(monkeypatch):
    monkeypatch.setattr(helper, "is_owned", lambda state: True)
    monkeypatch.setattr(helper.time, "sleep", lambda seconds: pytest.fail("No auth retry"))

    def fail(url):
        raise HTTPError(url, 401, "Unauthorized", {}, io.BytesIO(b"private-error-body"))

    monkeypatch.setattr(helper, "request_json", fail)
    with pytest.raises(RuntimeError, match="authentication failed") as caught:
        helper.wait_ready(saved_state(), 60)
    assert "private-error-body" not in str(caught.value)


def test_changed_managed_settings_require_explicit_stop(monkeypatch, tmp_path):
    monkeypatch.setattr(helper, "require_platform", lambda: None)
    monkeypatch.setattr(helper, "validate_environment", lambda directory: None)
    monkeypatch.setattr(helper, "read_state", lambda path: saved_state(command=["old-command"], gpu_index=0))
    monkeypatch.setattr(helper, "is_owned", lambda state: True)
    monkeypatch.setattr(helper.subprocess, "Popen", lambda *a, **kw: pytest.fail("No new server may start"))
    with pytest.raises(RuntimeError, match="different settings"):
        helper.start(arguments(tmp_path))


@pytest.mark.parametrize("failure_stage", ["compatibility", "readiness"])
def test_failed_start_stops_only_the_new_owned_server(monkeypatch, tmp_path, failure_stage):
    monkeypatch.setattr(helper, "require_platform", lambda: None)
    monkeypatch.setattr(helper, "validate_environment", lambda directory: None)
    monkeypatch.setattr(helper, "ROOT", tmp_path)
    monkeypatch.setattr(helper, "read_state", lambda path: None)
    monkeypatch.setattr(helper, "gpu_preflight", lambda index: {"uuid": "GPU-test", "compute_capability": 7.5})
    monkeypatch.setattr(helper, "process_identity", lambda pid: ("123456", "boot-id"))
    prepared = []

    def prepare(env_dir, gpu, environment):
        assert environment["CUDA_VISIBLE_DEVICES"] == gpu["uuid"]
        if failure_stage == "compatibility":
            raise RuntimeError("compatibility failed")
        prepared.append(env_dir)

    monkeypatch.setattr(helper, "prepare_attention_compat", prepare)

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def bind(self, address):
            assert address == ("127.0.0.1", 8001)

    monkeypatch.setattr(helper.socket, "socket", Socket)

    def spawn(command, **kwargs):
        assert prepared == [tmp_path / "server-venv"]
        assert kwargs["start_new_session"] is True
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-test"
        return SimpleNamespace(pid=12001)

    monkeypatch.setattr(helper.subprocess, "Popen", spawn)

    def fail(state, timeout):
        raise RuntimeError("readiness failed")

    stopped = []
    monkeypatch.setattr(helper, "wait_ready", fail)
    monkeypatch.setattr(helper, "stop_owned", lambda path, state: stopped.append(state))
    helper_args = arguments(tmp_path)
    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        helper.start(helper_args)
    if failure_stage == "readiness":
        assert len(stopped) == 1 and stopped[0]["pid"] == 12001
    else:
        assert stopped == []  # No process was created when the patch could not be applied.
    assert helper.read_state(helper_args.state) is None  # read_state is deliberately faked above.
