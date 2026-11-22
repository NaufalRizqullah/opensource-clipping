"""Small structured vLLM requests for selecting and describing one highlight."""

from .base import ProviderConfigError
from .chunking import plan_windows


PINNED_FIELDS = {"rank", "viral_score", "start_time", "end_time"}
EDIT_FIELDS = {
    "hook_start_time", "hook_end_time", "bgm_mood", "typography_plan",
    "broll_list", "recommended_visual_broll_hook", "hook_v2", "keep_segments",
}


def one_clip_schema(properties):
    item = {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}
    return {"type": "object", "required": ["clips"], "additionalProperties": False,
            "properties": {"clips": {"type": "array", "items": item, "minItems": 1, "maxItems": 1}}}


def metadata_fields(fields, *, editing):
    group = {name: field for name, field in fields.items()
             if name not in PINNED_FIELDS and (name in EDIT_FIELDS) == editing}
    # Encode the existing prompt's list limits in the decoder grammar as well.
    limits = {"typography_plan": 6, "broll_list": 3, "recommended_visual_broll_hook": 5, "keyword_tags": 8}
    for name in limits.keys() & group.keys():
        limit = min(limits[name], group[name].get("maxItems", limits[name]))
        group[name] = group[name] | {"maxItems": limit}
    return group


def choices_for(window, min_duration, max_duration):
    """Offer overlapping caption-aligned cuts, covering even the last caption."""
    preferred = min(max_duration, max(60, min_duration))
    for ceiling in dict.fromkeys((preferred, max_duration)):
        try:
            return plan_windows(
                window.text, lambda clip: clip.end - clip.start <= ceiling,
                min_duration=min_duration, overlap_seconds=ceiling / 2,
            )
        except ProviderConfigError:
            # Coarse captions may need a longer clip than the one-minute target.
            continue
    return []


def selection_messages(window, options):
    menu = "\n".join(f"{i}: [{option.start} - {option.end}]" for i, option in enumerate(options, 1))
    return [
        {"role": "system", "content": (
            "Pick the strongest standalone highlight from the numbered intervals below. "
            "Read the whole transcript section. Prefer a clear hook, a coherent point and a payoff. "
            "Rate this clip on a consistent 1-100 scale: 90-100 is exceptional, 70-89 strong, "
            "below 70 ordinary. Return only a JSON object with one clips item containing "
            "its choice number and viral_score. Intervals in absolute seconds:\n" + menu
        )},
        {"role": "user", "content": window.text},
    ]


def metadata_messages(instructions, excerpt, properties, *, editing, cfg):
    # The full worked example carries unrelated timestamps. Preserve the application's
    # field rules, languages and account mappings for the actual selected excerpt.
    rules = instructions.partition("\nSTRUKTUR JSON WAJIB")[0]
    classification = rules.partition("KLASIFIKASI AKUN TUJUAN")[2].partition("HOOK (WAJIB):")[0]
    editing_rules = rules.partition("HOOK (WAJIB):")[2].partition("SLOW CLOSING:")[0]
    output_rules = rules.partition("ATURAN OUTPUT:")[2]
    publishing_rules = rules.partition("ALASAN PEMILIHAN:")[2].partition("ATURAN OUTPUT:")[0]
    if editing_rules and publishing_rules:
        rules = editing_rules + "\n" + output_rules if editing else classification + "\n" + publishing_rules
    rules = rules.replace("Output HARUS berupa JSON array valid.",
                          'Output HARUS berupa objek JSON dengan satu field "clips" berisi tepat satu item.')
    task = (
        "Use absolute seconds inside the selected clip for all hooks, B-roll and keep segments. "
        "Order keep and B-roll segments by time. Use empty arrays when a feature is unused."
        if editing else "Write grounded publication metadata. Keep descriptions short and follow the field languages."
    )
    if editing and not getattr(cfg, "use_broll", True):
        task += " The editor disabled B-roll. Return [] for broll_list."
    return [
        {"role": "system", "content": (
            f"Selected clip: [{excerpt.start} - {excerpt.end}] seconds. Its boundaries are fixed. "
            f"{task} Return one clips item with only these fields: {', '.join(properties)}. "
            "Follow the supplied JSON schema. Keep array entries concise and avoid repeating items."
        )},
        {"role": "user", "content": rules + "\nTranskrip:\n" + excerpt.text},
    ]
