# Python dependencies

The base `requirements.txt` stays at the repository root. Specialized pip profiles and Kaggle constraints live in this directory. Run installation commands from the repository root.

| File | Purpose |
| --- | --- |
| [../requirements.txt](../requirements.txt) | Base CLI/application dependencies at the repository root |
| [requirements-kaggle.txt](requirements-kaggle.txt) | Isolated Kaggle Studio application with the shared constraints |
| [requirements-kaggle-cpu.txt](requirements-kaggle-cpu.txt) | Kaggle application with the matching CPU Torch packages |
| [requirements-test.txt](requirements-test.txt) | Lightweight dependencies for offline regression tests |
| [requirements-vllm-kaggle.txt](requirements-vllm-kaggle.txt) | Pinned vLLM server dependencies, installed in a separate server environment |
| [constraints-kaggle.txt](constraints-kaggle.txt) | Compatible package versions used by the Kaggle application profiles |

General installation:

```bash
python -m pip install -r requirements.txt
```

Offline test dependencies, in a separate development environment:

```bash
python -m pip install -r requirements/requirements-test.txt
```

Use the existing [Studio](../docs/KAGGLE_STUDIO.md) and [vLLM](../docs/KAGGLE_VLLM.md) setup helpers for their isolated environments. The constraints file restricts versions; it is not an installation list by itself.

The `-r` and `-c` includes inside these manifests are relative to the containing file. The Kaggle profile uses `-r ../requirements.txt` for the base dependencies and sibling filenames for profiles/constraints in this directory. Commands and scripts invoked from the repository root use the `requirements/` prefix for specialized profiles. `pyproject.toml` also remains at the repository root for packaging and `uv`.
