"""Offline patch/hardware-selection regressions; execute in Kaggle, not the PC.

Only temporary text files and fake device metadata are used. These tests never
import vLLM/Torch, initialize CUDA, install packages or load a model.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from scripts import kaggle_vllm_compat as compat


def original_source():
    return "# Unrelated module content must survive.\n" + compat.ORIGINAL_SELECTOR + "\nSENTINEL = 42\n"


def test_patch_is_idempotent_and_preserves_unrelated_source():
    source = original_source()
    patched = compat.patch_source(source, "0.32.0+cu129")
    assert patched != source
    assert compat.patch_source(patched, "0.32.0+cu129") == patched
    assert patched.replace(compat.PATCHED_SELECTOR, compat.ORIGINAL_SELECTOR) == source


@pytest.mark.parametrize("version", ["0.31.0", "0.32.0rc1", "0.33.0", "0.32.0+unknown"])
def test_unreviewed_versions_are_rejected_even_if_already_patched(version):
    for source in (original_source(), compat.PATCHED_SELECTOR):
        with pytest.raises(RuntimeError, match="supports only pinned"):
            compat.patch_source(source, version)


@pytest.mark.parametrize("source", [
    "def _get_tile_size(): return 64\n",
    compat.ORIGINAL_SELECTOR * 2,
    compat.PATCHED_SELECTOR.replace("return 16", "return 32"),
    compat.ORIGINAL_SELECTOR + "\n# " + compat.PATCH_MARKER,
])
def test_unknown_or_partially_patched_source_is_rejected(source):
    with pytest.raises(RuntimeError):
        compat.patch_source(source, "0.32.0")


@pytest.mark.parametrize("head,window,element,prefill,cuda,shared,expected", [
    (512, 0, 2, True, True, 65536, 16),  # Failing Gemma 4 global prefill on T4.
    (512, 0, 2, False, True, 65536, 16),  # Decode stays within the same limit.
    (384, 0, 2, True, True, 65536, 16),  # Triton pads the head dimension to 512.
    (512, 0, 2, True, True, 98304, 32),  # Hardware that fits the original tile.
    (512, 0, 2, True, True, 163840, 32),
    (256, 1024, 2, False, True, 65536, 32),  # Preserve Gemma 3 decode tuning.
    (256, 512, 2, True, True, 65536, 32),  # Preserve smaller heads.
    (128, 0, 2, False, True, 65536, 16),
    (512, 0, 1, True, True, 65536, 32),  # FP8 requires tile >= 32.
    (512, 0, 1, False, True, 65536, 32),
    (512, 0, 2, True, False, 65536, 32),  # No change to non-CUDA platforms.
])
def test_patched_selector_respects_shared_memory_and_dtype(
    monkeypatch, head, window, element, prefill, cuda, shared, expected
):
    for name in ("vllm", "vllm.utils", "vllm.utils.mem_utils"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    queries = []

    def shared_memory():
        queries.append(True)
        return shared

    sys.modules["vllm.utils.mem_utils"].get_max_shared_memory_bytes = shared_memory
    namespace = {
        "triton": SimpleNamespace(next_power_of_2=lambda size: 1 << (size - 1).bit_length()),
        "current_platform": SimpleNamespace(is_cuda=lambda: cuda),
        "_is_gemma3_attention": lambda size, sliding: sliding == 1024 and size in (128, 256),
    }
    exec(compile(compat.patch_source(original_source(), "0.32.0"), "<fake-vllm>", "exec"), namespace)
    assert namespace["_get_tile_size"](head, window, element, prefill) == expected
    assert bool(queries) == (element >= 2 and head > 256 and cuda)
    assert namespace["SENTINEL"] == 42


@pytest.fixture
def installation(monkeypatch, tmp_path):
    env_dir = tmp_path / "server-venv"
    source = env_dir / "lib/python3.12/site-packages" / compat.TARGET_FILE
    source.parent.mkdir(parents=True)
    source.write_text(original_source(), encoding="utf-8", newline="\n")
    monkeypatch.setattr(compat.sys, "prefix", str(env_dir))
    monkeypatch.setattr(compat.sys, "base_prefix", str(tmp_path / "bootstrap"))
    monkeypatch.setattr(compat.metadata, "distribution", lambda name: SimpleNamespace(
        version="0.32.0+cu129", locate_file=lambda path: source,
    ))
    return env_dir, source


def test_installation_backup_and_rerun_preserve_original_bytes(installation):
    env_dir, source = installation
    original = source.read_bytes()
    assert compat.patch_installation(env_dir) is True
    backup = source.with_name(source.name + ".osc-t4-original")
    assert backup.read_bytes() == original
    patched = source.read_bytes()
    assert compat.patch_installation(env_dir) is False
    assert source.read_bytes() == patched and backup.read_bytes() == original


def test_patcher_refuses_system_or_different_environment(monkeypatch, installation, tmp_path):
    env_dir, source = installation
    different = tmp_path / "application-venv"
    different.mkdir()
    with pytest.raises(RuntimeError, match="isolated vLLM environment"):
        compat.patch_installation(different)
    monkeypatch.setattr(compat.sys, "base_prefix", str(env_dir))
    with pytest.raises(RuntimeError, match="isolated vLLM environment"):
        compat.patch_installation(env_dir)
    assert source.read_text(encoding="utf-8") == original_source()


def test_distribution_outside_selected_environment_is_never_modified(monkeypatch, installation, tmp_path):
    env_dir, _ = installation
    external = tmp_path / "external-vllm.py"
    external.write_text(original_source(), encoding="utf-8", newline="\n")
    monkeypatch.setattr(compat.metadata, "distribution", lambda name: SimpleNamespace(
        version="0.32.0+cu129", locate_file=lambda path: external,
    ))
    with pytest.raises(RuntimeError, match="outside the selected environment"):
        compat.patch_installation(env_dir)
    assert external.read_text(encoding="utf-8") == original_source()


def test_conflicting_backup_is_not_overwritten(installation):
    env_dir, source = installation
    original = source.read_bytes()
    backup = source.with_name(source.name + ".osc-t4-original")
    backup.write_bytes(b"previous source requiring manual inspection")
    with pytest.raises(RuntimeError, match="backup differs"):
        compat.patch_installation(env_dir)
    assert source.read_bytes() == original
    assert backup.read_bytes() == b"previous source requiring manual inspection"


def test_failed_atomic_replace_keeps_original_and_backup(monkeypatch, installation):
    env_dir, source = installation
    original = source.read_bytes()

    def fail_replace(*args):
        raise OSError("replacement failed")

    monkeypatch.setattr(compat.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        compat.patch_installation(env_dir)
    assert source.read_bytes() == original
    assert source.with_name(source.name + ".osc-t4-original").read_bytes() == original
    assert list(source.parent.glob(".osc-t4-*")) == []
