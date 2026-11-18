"""Apply the bounded T4 attention workaround inside the Kaggle server venv.

No Torch/vLLM imports, model loading or downloads occur in this patcher. The
launcher invokes it with the server's Python before starting a Turing GPU.
The tile-size guard follows https://github.com/vllm-project/vllm/pull/39018;
it remains a workaround until verified on the target Kaggle runtime.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import os
from pathlib import Path
import sys
import tempfile


SUPPORTED_VERSIONS = {"0.32.0", "0.32.0+cu129"}
TARGET_FILE = "vllm/v1/attention/ops/triton_unified_attention.py"
PATCH_MARKER = "OSC_T4_ATTENTION_COMPAT_V1"

# SPDX-License-Identifier: Apache-2.0
# Original selector: Copyright contributors to the vLLM project.
# https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/ops/triton_unified_attention.py
# Require this exact function before editing a third-party installation.
ORIGINAL_SELECTOR = '''def _get_tile_size(
    head_size: int,
    sliding_window: int,
    element_size: int,
    is_prefill: bool,
) -> int:
    """Select tile size with Gemma3-specific optimization."""
    if _is_gemma3_attention(head_size, sliding_window):
        # Gemma3: use 32 for decode (default is 16)
        return 32

    # Default behavior
    if is_prefill:
        return 32
    # Note: tile size must be at least 32 for fp8 (element_size == 1).
    return 16 if element_size >= 2 else 32
'''

TILE_GUARD = '''    # OSC_T4_ATTENTION_COMPAT_V1: large FP16 heads need smaller tiles on T4.
    if (
        element_size >= 2
        and triton.next_power_of_2(head_size) >= 512
        and current_platform.is_cuda()
    ):
        from vllm.utils.mem_utils import get_max_shared_memory_bytes

        if get_max_shared_memory_bytes() < 98304:
            return 16
'''

PATCHED_SELECTOR = ORIGINAL_SELECTOR.replace(
    "    if _is_gemma3_attention", TILE_GUARD + "    if _is_gemma3_attention", 1
)


def patch_source(source: str, version: str) -> str:
    """Fail closed on version/source drift; applying our exact patch is idempotent."""
    if version not in SUPPORTED_VERSIONS:
        raise RuntimeError(
            f"T4 attention workaround supports only pinned vLLM 0.32.0 (found {version}). "
            "Review the workaround before changing the server version."
        )
    if PATCH_MARKER in source:
        if source.count(PATCHED_SELECTOR) == 1 and ORIGINAL_SELECTOR not in source:
            return source
        raise RuntimeError("T4 attention workaround is incomplete or modified; refusing to overwrite it.")
    if source.count(ORIGINAL_SELECTOR) != 1:
        raise RuntimeError(
            "vLLM tile selector differs from the pinned source; no files were changed. "
            "Inspect the installed server version before applying a different patch."
        )
    return source.replace(ORIGINAL_SELECTOR, PATCHED_SELECTOR, 1)


def patch_installation(env_dir: Path) -> bool:
    """Back up and atomically patch only the active, explicitly selected venv."""
    environment = env_dir.resolve(strict=True)
    if environment != Path(sys.prefix).resolve() or environment == Path(sys.base_prefix).resolve():
        raise RuntimeError("Run this helper with the isolated vLLM environment's Python and matching --venv.")
    distribution = metadata.distribution("vllm")
    target = Path(distribution.locate_file(TARGET_FILE)).resolve(strict=True)
    if not target.is_relative_to(environment) or not target.is_file():
        raise RuntimeError("vLLM source is outside the selected environment; refusing to modify it.")
    original = target.read_bytes()
    updated = patch_source(original.decode("utf-8"), distribution.version).encode("utf-8")
    if updated == original:
        return False
    # Syntax-check the proposed source without importing or executing the ML stack.
    compile(updated, str(target), "exec")
    backup = target.with_name(target.name + ".osc-t4-original")
    if backup.is_symlink():
        raise RuntimeError("T4 attention backup is a symlink; refusing to overwrite it.")
    try:
        with backup.open("xb") as output:
            output.write(original)
    except FileExistsError:
        if backup.read_bytes() != original:
            raise RuntimeError("Existing T4 attention backup differs; preserve it and inspect the installation.") from None
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".osc-t4-", delete=False) as output:
            temporary = Path(output.name)
            output.write(updated)
        os.chmod(temporary, target.stat().st_mode)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True)
    args = parser.parse_args(argv)
    changed = patch_installation(args.venv)
    print(
        f"T4 attention workaround {'applied' if changed else 'already applied'}: "
        "tile 16 for large FP16 heads on CUDA devices with <96 KiB shared memory. "
        "Run the Kaggle eval to verify inference."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, SyntaxError, metadata.PackageNotFoundError) as error:
        print(f"vLLM compatibility: {error}", file=sys.stderr)
        raise SystemExit(1)
