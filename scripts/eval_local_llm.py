"""Opt-in vLLM transcript smoke/evaluation. Run in the Kaggle app environment.

No model is loaded by this client; the configured server performs inference.
The JSON report includes generated clips. Console output contains counts only.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    from clipping.providers.config import vllm_config_values

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--clips", type=int, default=1)
    parser.add_argument("--model", dest="vllm_model")
    parser.add_argument("--base-url", dest="vllm_base_url")
    parser.add_argument("--max-tokens", dest="local_max_tokens", type=int)
    parser.add_argument("--context-limit", dest="vllm_context_limit", type=int)
    parser.add_argument("--timeout", dest="vllm_timeout", type=float)
    parser.add_argument("--output", type=Path, default=Path("outputs/vllm-eval.json"))
    args = parser.parse_args(argv)
    if not 1 <= args.clips <= 30:
        parser.error("--clips must be between 1 and 30")
    cfg = SimpleNamespace(
        ai_provider="vllm", jumlah_clip=args.clips, durasi_hook=3,
        hook_v2=False, no_segment_trim=False, silence_trim=False,
        **vllm_config_values(vars(args)),
    )
    report = {"provider": "vllm", "model": cfg.vllm_model, "requested_clips": args.clips, "success": False}
    try:
        # The application environment includes these imports; none loads a model.
        from clipping.engine import analyze_with_vllm
        from clipping.metadata import normalize_and_validate
        transcript = args.transcript.read_text(encoding="utf-8")
        clips = analyze_with_vllm(transcript, cfg)
        # The normalizer can print diagnostics; keep this console report bounded.
        metadata_report = []
        with redirect_stdout(io.StringIO()):
            normalize_and_validate(deepcopy(clips), report=metadata_report)
        warnings = sum(item["warnings"] != "OK" for item in metadata_report)
        report.update(
            success=True, returned_clips=len(clips),
            clips_with_metadata_warnings=warnings, clips=clips,
            run=getattr(cfg, "ai_run_meta", {}),
            note="Schema/timing validated; narrative and language quality still need human review.",
        )
    except Exception as exc:
        from clipping.providers.base import ProviderError
        # Unexpected errors may contain paths, response text or credentials.
        report.update(error_type=type(exc).__name__, run=getattr(cfg, "ai_run_meta", {}))
        report["error"] = str(exc) if isinstance(exc, ProviderError) else "Evaluation failed; inspect the Kaggle environment and input file."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"clips", "run"}}, ensure_ascii=False))
    print(f"Report: {args.output}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
