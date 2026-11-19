"""Command-only model routing coverage; run in Kaggle, not on the user's PC."""

from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest

from scripts import kaggle_vllm as helper


def arguments(tmp_path, **overrides):
    # Deliberately omit reasoning_parser to cover existing callers/fixtures.
    values = dict(venv=tmp_path / "server-venv", model=helper.DEFAULT_MODEL,
                  revision=None, port=8001, context=8192, max_num_seqs=1,
                  gpu_memory_utilization=0.85)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture(autouse=True)
def prevent_runtime_operations(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Model command tests must not start processes or contact a server")

    monkeypatch.setattr(helper.subprocess, "run", unexpected)
    monkeypatch.setattr(helper.subprocess, "Popen", unexpected)
    monkeypatch.setattr(helper, "request_json", unexpected)
    monkeypatch.setattr(helper, "gpu_preflight", unexpected)


@pytest.mark.parametrize("model,expected", [
    (helper.DEFAULT_MODEL, "gemma4"),
    ("google/gemma-4-E4B-it-qat-w4a16-ct", "gemma4"),
    ("Qwen/Qwen3-4B-AWQ", "qwen3"),
    ("Qwen/Qwen3-8B-AWQ", "qwen3"),
    ("Qwen/Qwen3-30B-A3B", "qwen3"),
    ("Qwen/Qwen3.5-4B", None),
    ("Qwen/Qwen3-Next-80B-A3B-Instruct", None),
    ("Qwen/Qwen3-VL-4B-Instruct", None),
    ("Qwen/Qwen3-", None),
    ("owner/Qwen3-4B-AWQ", None),
    ("owner/gemma-4-custom", None),
    ("served-alias", None),
])
def test_model_family_selects_parser_without_assuming_custom_aliases(tmp_path, model, expected):
    command = helper.build_server_command(arguments(tmp_path, model=model))
    assert command[command.index("serve") + 1] == model
    if expected is None:
        assert "--reasoning-parser" not in command
    else:
        assert command[command.index("--reasoning-parser") + 1] == expected
    assert json.loads(command[command.index("--default-chat-template-kwargs") + 1]) == {
        "enable_thinking": False,
    }
    assert "--quantization" not in command


@pytest.mark.parametrize("model,revision,expected", [
    (helper.DEFAULT_MODEL, None, helper.DEFAULT_REVISION),
    (helper.DEFAULT_MODEL, "explicit-default-revision", "explicit-default-revision"),
    ("Qwen/Qwen3-4B-AWQ", None, None),
    ("Qwen/Qwen3-4B-AWQ", "explicit-qwen-revision", "explicit-qwen-revision"),
    ("owner/custom-model", "custom-revision", "custom-revision"),
])
def test_model_routing_preserves_checkpoint_revision(tmp_path, model, revision, expected):
    command = helper.build_server_command(arguments(tmp_path, model=model, revision=revision))
    for flag in ("--revision", "--tokenizer-revision"):
        if expected is None:
            assert flag not in command
        else:
            assert command[command.index(flag) + 1] == expected


@pytest.mark.parametrize("model,parser,expected", [
    ("Qwen/Qwen3-4B-AWQ", "auto", "qwen3"),
    (helper.DEFAULT_MODEL, "none", None),
    ("Qwen/Qwen3-4B-AWQ", "none", None),
    ("owner/custom-model", "qwen3", "qwen3"),
    (helper.DEFAULT_MODEL, "custom_plugin_parser", "custom_plugin_parser"),
])
def test_reasoning_parser_cli_reaches_start(monkeypatch, tmp_path, model, parser, expected):
    commands = []
    monkeypatch.setattr(helper, "require_platform", lambda: None)
    monkeypatch.setattr(helper, "state_lock", lambda path: nullcontext())
    monkeypatch.setattr(helper, "start", lambda args: commands.append(helper.build_server_command(args)))

    assert helper.main(["start", "--venv", str(tmp_path / "server-venv"),
                        "--model", model, "--reasoning-parser", parser]) == 0

    assert len(commands) == 1
    command = commands[0]
    if expected is None:
        assert "--reasoning-parser" not in command
    else:
        assert command[command.index("--reasoning-parser") + 1] == expected


def test_cli_defaults_to_auto_for_qwen(monkeypatch):
    received = []
    monkeypatch.setattr(helper, "require_platform", lambda: None)
    monkeypatch.setattr(helper, "state_lock", lambda path: nullcontext())
    monkeypatch.setattr(helper, "start", lambda args: received.append(args.reasoning_parser))

    assert helper.main(["start", "--model", "Qwen/Qwen3-4B-AWQ"]) == 0
    assert received == ["auto"]
