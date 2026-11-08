"""Keep API regression tests away from user credentials and persisted jobs."""

import os
from pathlib import Path
import tempfile

import pytest

_test_state = tempfile.TemporaryDirectory(prefix="osc-tests-")
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["OSC_JOB_STORE_PATH"] = str(Path(_test_state.name) / "jobs.json")


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_state():
    yield
    _test_state.cleanup()
