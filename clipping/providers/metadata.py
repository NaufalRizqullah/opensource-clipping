"""Small, atomic run provenance files separate from the legacy response cache."""

import hashlib
import json
from pathlib import Path


def _response_hash(path):
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def save_run_metadata(cfg, *, cached=False):
    path = Path(cfg.outputs_dir) / "ai_run_meta.json"
    response_path = path.with_name("gemini_response.json")
    if cached:
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(previous, dict):
                previous = {}
        except (OSError, ValueError):
            previous = {}
        response_hash = _response_hash(response_path)
        matches = (
            response_hash is not None
            and previous.get("status") == "completed"
            and previous.get("response_sha256") == response_hash
        )
        meta = {
            # A failed rerun can leave an older response file behind. Attribute
            # only a completed run whose exact cached bytes still match.
            "actual_provider": previous.get("actual_provider") if matches else None,
            "model": previous.get("model") if matches else None,
            "status": "completed",
            "cache_hit": True,
            "attempts": 0,
            "duration_seconds": 0,
        }
        if response_hash is not None:
            meta["response_sha256"] = response_hash
    else:
        meta = dict(getattr(cfg, "ai_run_meta", None) or {
            # Legacy providers may fallback internally; do not invent provenance.
            "actual_provider": None, "model": None, "instrumented": False,
            "status": "completed",
        })
        meta["cache_hit"] = False
        meta.pop("response_sha256", None)
        if meta.get("status") == "completed":
            response_hash = _response_hash(response_path)
            if response_hash is not None:
                meta["response_sha256"] = response_hash
    meta["requested_provider"] = getattr(cfg, "ai_provider", "gemini")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
