"""
clipping.story.loader - JSON Parser & Validator for Story Clip


  - sources.json  → source video registry
  - story_recipe.json → story recipe (clip, scene, hook, highlight)
"""

import json
import os
from types import SimpleNamespace


# ==============================================================================
# SCHEMA CONSTANTS
# ==============================================================================

SUPPORTED_PLATFORMS = {"youtube", "tiktok", "instagram", "gdrive", "local"}

_REQUIRED_SOURCE_FIELDS = {"id", "name", "platform"}
_REQUIRED_CLIP_FIELDS = {"clip_id", "title", "hook", "highlight"}
_REQUIRED_SCENE_FIELDS = {"source_id"}


# ==============================================================================
# SOURCES.JSON LOADER
# ==============================================================================

def load_sources(path: str) -> dict[str, dict]:
    """
    Parse and validate a ``sources.json`` file.

    Parameters
    ----------
    path : str
        Absolute or relative path to the sources JSON file.

    Returns
    -------
    dict[str, dict]
        Mapping of ``source_id`` → source entry dict.  Each entry
        contains at minimum: ``id``, ``name``, ``platform``, and either
        ``url`` or ``local_path``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If schema validation fails.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sources file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    entries = raw.get("sources", [])
    if not entries:
        raise ValueError(
            f"sources.json is empty or missing the 'sources' key: {path}"
        )

    registry: dict[str, dict] = {}

    for idx, src in enumerate(entries):
        # --- Required fields ---
        missing = _REQUIRED_SOURCE_FIELDS - set(src.keys())
        if missing:
            raise ValueError(
                f"Source #{idx} ('{src.get('id', '?')}') is missing required field(s): {missing}"
            )

        sid = src["id"]
        platform = src["platform"]

        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Source '{sid}': platform '{platform}' is not recognized. "
                f"Choose from: {SUPPORTED_PLATFORMS}"
            )

        # --- URL / local_path check ---
        url = src.get("url")
        local_path = src.get("local_path")

        if platform == "local":
            if not local_path:
                raise ValueError(
                    f"Source '{sid}': platform 'local' requires 'local_path'."
                )
            if not os.path.exists(local_path):
                raise ValueError(
                    f"Source '{sid}': local_path not found: {local_path}"
                )
        else:
            if not url:
                raise ValueError(
                    f"Source '{sid}': platform '{platform}' requires 'url'."
                )

        # --- Duplicate check ---
        if sid in registry:
            raise ValueError(f"Duplicate source id: '{sid}'")

        registry[sid] = src

    print(f"✅ Loaded {len(registry)} source(s) from {os.path.basename(path)}")
    return registry


# ==============================================================================
# STORY_RECIPE.JSON LOADER
# ==============================================================================

def _validate_scene(scene: dict, source_registry: dict, clip_id: int, section: str, idx: int) -> None:
    """Validate a single scene entry within a clip."""
    missing = _REQUIRED_SCENE_FIELDS - set(scene.keys())
    if missing:
        raise ValueError(
            f"Clip #{clip_id} → {section} → scene #{idx}: missing field(s) {missing}"
        )

    sid = scene["source_id"]
    if sid not in source_registry:
        raise ValueError(
            f"Clip #{clip_id} → {section} → scene #{idx}: "
            f"source_id '{sid}' not found in sources.json. "
            f"Available IDs: {list(source_registry.keys())}"
        )

    start = scene.get("start")
    end = scene.get("end")

    # Allow null timestamps (scene will be skipped during render)
    if start is not None and end is not None:
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise ValueError(
                f"Clip #{clip_id} → {section} → scene #{idx}: "
                f"start/end must be a number or null."
            )
        if end <= start:
            raise ValueError(
                f"Clip #{clip_id} → {section} → scene #{idx}: "
                f"end ({end}) must be greater than start ({start})."
            )


def load_recipe(path: str, source_registry: dict[str, dict]) -> dict:
    """
    Parse and validate a ``story_recipe.json`` file.

    Parameters
    ----------
    path : str
        Path to the recipe JSON.
    source_registry : dict
        The registry dict returned by :func:`load_sources`.

    Returns
    -------
    dict
        The full validated recipe dict with a ``clips`` list and
        ``default_settings``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If schema validation fails.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Recipe file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        recipe = json.load(f)

    clips = recipe.get("clips", [])
    if not clips:
        raise ValueError(
            f"story_recipe.json is empty or missing the 'clips' key: {path}"
        )

    seen_ids: set[int] = set()

    for clip in clips:
        # --- Required clip fields ---
        missing = _REQUIRED_CLIP_FIELDS - set(clip.keys())
        if missing:
            raise ValueError(
                f"Clip (id={clip.get('clip_id', '?')}): missing field(s) {missing}"
            )

        cid = clip["clip_id"]
        if cid in seen_ids:
            raise ValueError(f"Duplicate clip_id: {cid}")
        seen_ids.add(cid)

        # --- Validate hook ---
        hook = clip["hook"]
        hook_scenes = hook.get("scenes", [])
        if not hook_scenes:
            raise ValueError(f"Clip #{cid}: hook.scenes is empty.")
        for i, scene in enumerate(hook_scenes):
            _validate_scene(scene, source_registry, cid, "hook", i)

        # --- Validate highlight ---
        highlight = clip["highlight"]
        hl_scenes = highlight.get("scenes", [])
        if not hl_scenes:
            raise ValueError(f"Clip #{cid}: highlight.scenes is empty.")
        for i, scene in enumerate(hl_scenes):
            _validate_scene(scene, source_registry, cid, "highlight", i)

    # Merge default_settings
    defaults = recipe.get("default_settings", {})
    recipe["_defaults"] = SimpleNamespace(
        ratio=defaults.get("ratio", "9:16"),
        font_style=defaults.get("font_style", "HORMOZI"),
        use_subtitle=defaults.get("use_subtitle", True),
        use_bgm=defaults.get("use_bgm", True),
        bgm_mood=defaults.get("bgm_mood", "upbeat"),
        min_duration=defaults.get("min_duration", 15),
    )

    print(
        f"✅ Loaded {len(clips)} clip(s) from {os.path.basename(path)} "
        f"(project: {recipe.get('project_name', 'Untitled')})"
    )
    return recipe


# ==============================================================================
# HELPER: RESOLVE SCENE → FILE PATH
# ==============================================================================

def resolve_scene_path(scene: dict, source_registry: dict, cache_dir: str) -> str | None:
    """
    Resolve a scene's source_id to the cached file path.

    Returns None if the scene has null timestamps (should be skipped).
    """
    if scene.get("start") is None or scene.get("end") is None:
        return None

    sid = scene["source_id"]
    src = source_registry[sid]

    if src["platform"] == "local":
        return src["local_path"]

    # For remote sources, expect the file in cache
    cached = os.path.join(cache_dir, f"{sid}.mp4")
    if not os.path.exists(cached):
        raise FileNotFoundError(
            f"Cached video for source '{sid}' not found at {cached}. "
            "Run the download first."
        )
    return cached
