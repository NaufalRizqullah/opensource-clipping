"""Storage layout regressions for execution in the hosted notebook."""

from types import SimpleNamespace

import pytest

from scripts import kaggle_storage as storage


@pytest.fixture
def project(tmp_path):
    checkout = tmp_path / "working" / "checkout"
    checkout.mkdir(parents=True)
    return checkout


def test_temporary_layout_reuses_paths_and_keeps_job_files(monkeypatch, tmp_path, project):
    monkeypatch.setattr(storage.tempfile, "gettempdir", lambda: str(tmp_path / "scratch"))
    saved = project / "outputs" / "job" / "clip.mp4"
    saved.parent.mkdir(parents=True)
    saved.touch()
    legacy = storage.prepare_storage(project, project.parent, mode="checkout")
    legacy.studio_env.mkdir(parents=True)
    first = storage.prepare_storage(project, project.parent)
    assert first == storage.prepare_storage(project, project.parent)
    assert first.root.is_relative_to(tmp_path / "scratch")
    assert first.studio_env != first.vllm_env
    assert all(path.is_relative_to(first.root) for path in (first.studio_env, first.vllm_env, first.python_cache))
    assert legacy.studio_env == project / ".cache/kaggle-studio/venv"
    assert legacy.studio_env.is_dir() and saved.is_file()


def test_output_location_cannot_be_used_as_scratch(tmp_path, project):
    with pytest.raises(ValueError, match="outside"):
        storage.prepare_storage(project, project.parent, runtime_root=project.parent / "runtime")


def test_output_reserve_is_separate_from_runtime_space(monkeypatch, tmp_path, project):
    layout = storage.prepare_storage(project, project.parent, runtime_root=tmp_path / "scratch")
    def usage(path):
        return SimpleNamespace(free=(10 if path == layout.root else 1) * storage.GIB)
    monkeypatch.setattr(storage.shutil, "disk_usage", usage)
    storage.check_storage(layout, min_output_gib=1)
    with pytest.raises(RuntimeError, match="Insufficient output space"):
        storage.check_storage(layout, min_output_gib=2)


def test_explicit_cache_locations_survive_temporary_storage(monkeypatch, tmp_path, project):
    layout = storage.prepare_storage(project, project.parent, runtime_root=tmp_path / "runtime")
    monkeypatch.setenv("HF_HOME", "/mounted/model-cache")
    monkeypatch.setenv("TORCH_HOME", "/mounted/torch-cache")
    assert layout.cache_environment() == {"TMPDIR": str(layout.root / "tmp")}
