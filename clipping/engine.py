"""
clipping.engine - Download, Transcription & Gemini AI Analysis

Maps to Cell 2 (The Engine) of the notebook.
"""

import json
import os
import re
import time

from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel


# ==============================================================================
# TAHAP 1: DOWNLOAD VIDEO
# ==============================================================================

def _build_ydl_format_selector(download_source_height: str | int) -> str:
    """
    Build a yt-dlp format selector string for source-quality preference.
    """
    # Skip AV1 codec as it lacks HW acceleration on many platforms (e.g., Colab T4)
    # and causes decoding failures in OpenCV/FFmpeg software fallbacks.
    # Note: Using [vcodec!^=av01] to match the start of the codec ID.
    codec_filter = "[vcodec!^=av01]"

    if download_source_height == "max":
        return f"bestvideo{codec_filter}+bestaudio/best{codec_filter}/best"

    try:
        h_val = int(download_source_height)
    except (ValueError, TypeError):
        h_val = 0

    if 0 < h_val <= 1080:
        # For standard resolutions, strictly prefer native MP4 (H.264/AAC)
        return (
            f"bestvideo[height<=?{h_val}][ext=mp4]+bestaudio[ext=m4a]/"
            f"bestvideo[height<=?{h_val}]{codec_filter}+bestaudio/"
            f"best[height<=?{h_val}][ext=mp4]/"
            f"best[height<=?{h_val}]{codec_filter}/"
            f"best"
        )

    return (
        f"bestvideo[height<=?{download_source_height}]{codec_filter}+bestaudio/"
        f"best[height<=?{download_source_height}]{codec_filter}/"
        f"best"
    )


_PLATFORM_LABELS = {
    "youtube": "YouTube",
    "tiktok": "TikTok",
    "instagram": "Instagram",
    "gdrive": "Google Drive",
}


def _extract_gdrive_file_id(url: str) -> str | None:
    """Extract the Google Drive file ID from various URL formats."""
    import re as _re
    m = _re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = _re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    return None


def _download_gdrive(url: str, output_path: str) -> None:
    """Download a video from Google Drive using gdown (more reliable than yt-dlp)."""
    import gdown

    file_id = _extract_gdrive_file_id(url)
    if not file_id:
        raise RuntimeError(
            f"Could not extract the file ID from the Google Drive URL: {url}\n"
            "      Supported formats:\n"
            "        • https://drive.google.com/file/d/FILE_ID/view\n"
            "        • https://drive.google.com/open?id=FILE_ID"
        )

    download_url = f"https://drive.google.com/uc?id={file_id}"
    print(f"      📥 File ID: {file_id}")
    gdown.download(download_url, output_path, quiet=False)


def download_video(
    url: str,
    output_path: str,
    use_dlp_subs: bool = False,
    download_source_height: str | int = "max",
    source_platform: str = "youtube",
) -> None:
    """
    Download a video to *output_path* with configurable source height.

    Parameters
    ----------
    source_platform : str
        One of ``"youtube"`` (default), ``"tiktok"``, ``"instagram"``,
        or ``"gdrive"``.
    """
    platform_label = _PLATFORM_LABELS.get(source_platform, source_platform)
    uses_youtube_format = source_platform == "youtube"

    print(f"[1/3] Downloading video from {platform_label}...")
    if download_source_height == "max":
        print("      🎯 Source quality: highest available", flush=True)
    else:
        print(f"      🎯 Source quality: up to {download_source_height}p", flush=True)

    # --- Google Drive: use gdown instead of yt-dlp ---
    if source_platform == "gdrive":
        _download_gdrive(url, output_path)
        if not os.path.exists(output_path):
            raise RuntimeError(
                f"❌ Google Drive download failed - file not found at {output_path}"
            )
        print(f"      ✅ Video successfully downloaded from Google Drive.", flush=True)
        return

    # --- Build yt-dlp options per platform ---
    if uses_youtube_format:
        # YouTube: complex format selector + AV1 filter + remote components
        ydl_opts = {
            "format": _build_ydl_format_selector(download_source_height),
            "outtmpl": output_path,
            "quiet": True,
            "merge_output_format": "mp4",
            "remote_components": ["ejs:github"],
        }
    else:
        # TikTok / Instagram: ensure video and audio are merged
        # We explicitly prefer H.264 over H.265 (TikTok's bytevc1) to prevent 
        # PyAV/faster-whisper from crashing with IndexError on Kaggle/Colab.
        ydl_opts = {
            "format": "bestvideo[vcodec^=h264]+bestaudio/best[vcodec^=h264]/best",
            "outtmpl": output_path,
            "quiet": True,
            "merge_output_format": "mp4",
        }

    # --- Subtitle download - only supported for YouTube ---
    if use_dlp_subs and uses_youtube_format:
        print("      Trying to find auto-generated subtitles (en / id)...")
        import glob

        for lang in ["en", "id"]:
            ydl_opts_subs = ydl_opts.copy()
            ydl_opts_subs.update({
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": [lang],
                "subtitlesformat": "json3",
                "skip_download": True,  # Only fetch the subtitle
            })

            try:
                with YoutubeDL(ydl_opts_subs) as ydl:
                    ydl.download([url])

                # Check whether the json3 subtitle for this language actually downloaded
                if glob.glob(output_path.replace(".mp4", f".*.json3")):
                    print(f"      ✅ Subtitle '{lang}' found. Continuing to video...")
                    break
            except Exception as e:
                print(f"      ⚠️ Failed to pull subtitle '{lang}' ({e}). Trying next option...")
    elif use_dlp_subs and not uses_youtube_format:
        print(f"      ℹ️ {platform_label} does not provide auto subtitles. Whisper will be used.")

    # Run the video download separately from the subtitle handling
    with YoutubeDL(ydl_opts) as ydl:
        # Extra step to verify resolution before downloading
        try:
            info = ydl.extract_info(url, download=False)
            best_h = info.get("height", "unknown")
            v_codec = info.get("vcodec", "unknown")
            print(f"      ✅ Downloading: {best_h}p (Codec: {v_codec})", flush=True)
        except Exception as e:
            print(f"      ⚠️ Failed to check detailed info: {e}", flush=True)

        ydl.download([url])

    # --- Post-download verification ---
    if not os.path.exists(output_path):
        raise RuntimeError(
            f"❌ {platform_label} download failed - video file not found at {output_path}.\n"
            "      Make sure the URL is valid and publicly accessible."
        )


# ==============================================================================
# TAHAP 2: TRANSKRIPSI WHISPER & JSON3 FALLBACK
# ==============================================================================

def parse_youtube_json3_subs(json_path: str, max_words_per_subtitle: int = 5) -> tuple[str, list[dict]]:
    """
    Parse downloaded YouTube JSON3 subtitles into transkrip_lengkap and data_segmen.
    Returns empty string/list if parsing fails.
    """
    import json

    print("[2/3] Processing JSON3 subtitles from YouTube...")
    transkrip_lengkap = ""
    data_segmen = []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            subs_data = json.load(f)

        events = subs_data.get("events", [])

        flat_words = []
        for event in events:
            # YouTube timestamps are in ms
            t_start = event.get("tStartMs", 0) / 1000.0
            d_duration = event.get("dDurationMs", 0) / 1000.0
            event_end = t_start + d_duration

            segs = event.get("segs", [])
            for i, seg in enumerate(segs):
                text = seg.get("utf8", "")
                if not text.strip() or text == "\n":
                    continue

                # tOffsetMs is offset from t_start
                offset = seg.get("tOffsetMs", 0) / 1000.0
                seg_start = t_start + offset

                # Determine end of this segment
                if i < len(segs) - 1:
                    next_offset = segs[i + 1].get("tOffsetMs", 0) / 1000.0
                    seg_end = t_start + next_offset
                else:
                    seg_end = event_end

                if seg_end <= seg_start:
                    seg_end = seg_start + 1.0  # Fallback duration

                # Clean up word text
                clean_text = text.replace("\n", " ").replace("\u200b", "").strip()
                clean_text = re.sub(r"[^\x00-\x7F\u00C0-\u017F\u2018-\u201F\u2026]", "", clean_text)

                if clean_text:
                    # Memecah teks menjadi kata tunggal agar karaoke per-kata bekerja seperti whisper
                    words_in_seg = clean_text.split()
                    if not words_in_seg:
                        continue

                    duration_per_word = (seg_end - seg_start) / len(words_in_seg)

                    for w_idx, w_text in enumerate(words_in_seg):
                        w_start = seg_start + (w_idx * duration_per_word)
                        w_end = w_start + duration_per_word

                        flat_words.append({
                            "word": w_text,
                            "start": w_start,
                            "end": w_end,
                        })

        # Adjust end times based on the start time of the next word to prevent overlaps
        for i in range(len(flat_words) - 1):
            if flat_words[i]["end"] > flat_words[i + 1]["start"]:
                flat_words[i]["end"] = max(flat_words[i]["start"] + 0.1, flat_words[i + 1]["start"])

        # Group them into segments
        chunk_words = []
        chunk_start = 0.0

        for i, w in enumerate(flat_words):
            if len(chunk_words) == 0:
                chunk_start = w["start"]

            chunk_words.append(w)

            if len(chunk_words) == max_words_per_subtitle or i == len(flat_words) - 1:
                chunk_text = " ".join([cw["word"] for cw in chunk_words])
                chunk_end = w["end"]
                transkrip_lengkap += f"[{chunk_start:.1f} - {chunk_end:.1f}] {chunk_text}\n"

                data_segmen.append({
                    "start": chunk_start,
                    "end": chunk_end,
                    "words": chunk_words,
                })
                chunk_words = []

        return transkrip_lengkap, data_segmen

    except Exception as e:
        print(f"⚠️ Failed to parse JSON3: {e}")
        return "", []


def transcribe_video(
    video_path: str,
    max_words_per_subtitle: int = 5,
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
) -> tuple[str, list[dict]]:
    """
    Transcribe *video_path* using Faster-Whisper.

    Returns
    -------
    transkrip_lengkap : str
        Human-readable transcript with timestamps.
    data_segmen : list[dict]
        Word-level segments grouped by *max_words_per_subtitle*.
    """
    print("[2/3] Starting transcription with Faster-Whisper (per-word level)...")

    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        # Covers no-CUDA hosts (macOS) AND float16-on-CPU (unsupported by ctranslate2).
        # int8 on CPU is the safe, broadly-supported combo.
        if (device, compute_type) == ("cpu", "int8"):
            raise
        print(f"[WARN] Whisper ({device}/{compute_type}) failed ({e}). Falling back to CPU/int8.")
        device = "cpu"
        compute_type = "int8"
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(video_path, beam_size=5, word_timestamps=True)

    transkrip_lengkap = ""
    data_segmen: list[dict] = []

    for segment in segments:
        transkrip_lengkap += f"[{segment.start:.1f} - {segment.end:.1f}] {segment.text}\n"

        if segment.words:
            chunk_words: list[dict] = []
            chunk_start = 0.0

            for i, w in enumerate(segment.words):
                if len(chunk_words) == 0:
                    chunk_start = w.start

                chunk_words.append({
                    "word": w.word.strip(),
                    "start": w.start,
                    "end": w.end,
                })

                if len(chunk_words) == max_words_per_subtitle or i == len(segment.words) - 1:
                    data_segmen.append({
                        "start": chunk_start,
                        "end": w.end,
                        "words": chunk_words,
                    })
                    chunk_words = []

    return transkrip_lengkap, data_segmen


# ==============================================================================
# STAGE 3: AI ANALYSIS
# ==============================================================================


# ---- Retry Config ----
MAX_ATTEMPTS = 10
INITIAL_WAIT_SECONDS = 60
WAIT_INCREMENT_SECONDS = 30
REQUEST_TIMEOUT_MS = 15 * 60 * 1000  # 15 menit
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _extract_status_code(exc: Exception):
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    match = re.search(r"\b(408|429|500|502|503|504)\b", str(exc))
    return int(match.group(1)) if match else None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, json.JSONDecodeError):
        return True

    code = _extract_status_code(exc)
    if code in RETRYABLE_STATUS_CODES:
        return True

    msg = str(exc).lower()
    keywords = (
        "timeout", "temporarily unavailable", "deadline",
        "connection reset", "connection aborted", "service unavailable",
    )
    return any(k in msg for k in keywords)


def _generate_json_with_retry(client, model, fallback_model, contents, config):
    last_exc = None
    status_code = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"[Gemini] Attempt {attempt}/{MAX_ATTEMPTS}...")

            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

            text = getattr(response, "text", None)
            if not text or not text.strip():
                raise ValueError("Gemini returned an empty response.text.")

            return json.loads(text)

        except Exception as exc:
            last_exc = exc
            status_code = _extract_status_code(exc)
            retryable = _is_retryable(exc)

            print(
                f"[Gemini] Attempt {attempt}/{MAX_ATTEMPTS} failed | "
                f"status={status_code} | error={exc}"
            )

            if (not retryable) or attempt == MAX_ATTEMPTS:
                break

            wait_seconds = INITIAL_WAIT_SECONDS + ((attempt - 1) * WAIT_INCREMENT_SECONDS)
            print(f"[Gemini] Retry lagi dalam {wait_seconds} detik...")
            time.sleep(wait_seconds)

    print(f"[Gemini] Attempt with primary model ({model}) failed.")
    if fallback_model:
        print(f"[Gemini] Trying once more with fallback model ({fallback_model})...")
        try:
            response = client.models.generate_content(
                model=fallback_model,
                contents=contents,
                config=config,
            )
            text = getattr(response, "text", None)
            if not text or not text.strip():
                raise ValueError("Gemini fallback returned an empty response.text.")

            return json.loads(text)
        except Exception as exc_fallback:
            print(f"[Gemini] Fallback model failed | error={exc_fallback}")
            raise RuntimeError(
                f"Failed to call primary & fallback Gemini. "
                f"Primary report status={status_code}, error={last_exc} | "
                f"Fallback report error={exc_fallback}"
            ) from exc_fallback

    raise RuntimeError(
        f"Failed to call Gemini after {MAX_ATTEMPTS} attempts. Last error: {last_exc}"
    ) from last_exc


def get_analysis_prompt(transkrip_lengkap: str, jumlah_clip: int, durasi_hook: int, cfg=None) -> str:
    """Centralized prompt for both Gemini and NVIDIA providers."""
    # Build optional Hook V2 prompt section
    _hook_v2_prompt = ""
    if cfg and getattr(cfg, "hook_v2", False):
        _hook_v2_items = getattr(cfg, "hook_v2_items", 3)
        _hook_v2_style = getattr(cfg, "hook_v2_style", "controversial_fast_glitch")
        _hook_v2_prompt = f"""

HOOK V2 (MULTI-HOOK INTRO - REQUIRED):
- In addition to the standard hook, create a "hook_v2" object with {_hook_v2_items} short cuts (0.5-2 seconds) taken from the most striking/controversial/emotional moments inside the clip.
- Style: {_hook_v2_style}
- Each item must contain: start_time, end_time, and text (a short 2-5 word on-screen caption).
- Order the items from strongest to weakest.
- Transitions between items are added automatically (white flash / glitch) by the system.
- Fill the "hook_v2" field as an object with:
  - "enabled": true
  - "items": an array of objects (start_time, end_time, text)
  - "transition": an object with "type" ("white_flash" or "glitch")
"""

    # Build optional Segment Trimming prompt section
    _segment_prompt = ""
    if cfg and not getattr(cfg, "no_segment_trim", False):
        _silence_hint = ""
        if cfg and getattr(cfg, "silence_trim", False):
            _silence_hint = "\n- AGGRESSIVELY remove silent/dead-air sections. Do not include pauses longer than 0.5 seconds."
        _segment_prompt = f"""

SEGMENT-BASED TRIMMING (KEEP SEGMENTS - REQUIRED):
- For each clip, analyze whether there are parts that are less interesting, too quiet, rambling, or filler in the middle.
- If so, split the clip into several "keep_segments" - keeping only the best parts.
- Each segment contains: start_time and end_time.
- Segments must be in chronological order and must not overlap.
- If the entire clip duration is already tight and engaging, just create a single segment covering the whole duration.{_silence_hint}
- Fill the "keep_segments" field as an array of objects (start_time, end_time).
"""
    return f"""
You are an Art Director, Video Editor, and Short-Form Content Metadata Strategist for TikTok, Reels, and YouTube Shorts.

Read the following video transcript. Transcript format:
[start_second - end_second] text

MAIN TASK:
- Find the {jumlah_clip} most interesting, strongest, most shareable, and most viral-worthy moments to turn into short clips.
- Order the clips by viral_score from highest (most likely to go viral) to lowest. The "rank" is only a sequential number (1, 2, 3...).
- For each clip, produce the clip timing, hook, typography plan, b-roll plan, selection reasoning, and cross-platform metadata.
- All output must be highly relevant to the clip's content, not the full video in general.

CLIP SELECTION & VIRALITY RULES:
- Clip duration must be 30-180 seconds.
- Choose parts that have emotion, conflict, surprise, insight, a strong opinion, a practical lesson, or a clear punchline.
- Evaluate virality and assign a "viral_score" (1-100) representing how viral a clip is.
  - 90-100: Very strong fyp/viral potential, strong emotion/conflict, a hook that really lands.
  - 80-89: Engaging, likely to perform well.
  - 70-79: Standard, informative but maybe lacking punch.
- Prefer parts that stay interesting even when watched without the full video's context.
- Avoid clips whose content is too similar to one another.
- Do not choose clips that feel flat, rambling, or that lack a clear payoff.

RETENTION & CLIP STRUCTURE RULES:
- Make sure the first 3 seconds have strong pull: a hook, conflict, curiosity, a sharp statement, emotion, or an implicit question.
- An ideal clip follows the structure:
  hook -> brief context -> tension/insight -> payoff.
- Do not choose clips that only get interesting after running too long.
- If the start of a segment is too slow, shift start_time to a stronger sentence.
- If the payoff is already over, do not extend the clip without reason.
- Do not include intros, small talk, long pauses, or transitions that add no appeal.
- Prefer clips that make the viewer want to:
  1. stop scrolling,
  2. watch to the end,
  3. comment,
  4. share,
  5. save,
  6. or feel "this is so me".

TIMING / CUT RULES:
- start_time must begin as close as possible to the first strong moment, not just the start of the topic.
- end_time must stop after the payoff, conclusion, punchline, or main emotional beat is finished.
- Do not cut too early if a sentence is still hanging.
- Do not let the clip run too long after the core message is done.
- The clip must remain understandable without watching what comes before or after it.
- If two strong moments are very close and support each other, you may merge them as long as the duration stays 30-180 seconds.
- If two strong moments have different angles, separate them as different clip candidates.

INTERNAL VIRAL_SCORE EVALUATION:
Score viral_score 1-100 based on the following components. This is for internal evaluation only - DO NOT add new fields to the JSON.
- Hook strength: 1-20
- Emotional intensity: 1-20
- Shareability/comment potential: 1-20
- Standalone clarity: 1-20
- Payoff/retention: 1-20

Scoring guide:
- Hook strength: how strongly the first 3 seconds make people stop scrolling.
- Emotional intensity: how strong the emotion, conflict, unease, humor, poignancy, anger, awe, or relatability is.
- Shareability/comment potential: how likely people are to comment, debate, tag a friend, share, or save.
- Standalone clarity: how easily the clip is understood without the full video's context.
- Payoff/retention: how clear the reward for watching to the end is, such as a punchline, insight, twist, conclusion, or practical lesson.
- Do not choose clips with a viral_score below 70 unless the number of good moments in the transcript is very limited.

HOOK (REQUIRED):
- Take the single punchiest sentence that EXISTS INSIDE the clip.
- The hook must feel strong and grab attention within the first ~{durasi_hook} seconds.
- Store it as hook_start_time and hook_end_time.
- The hook must make people want to keep watching, but no fake clickbait.
- Make sure the hook is natural and actually spoken in the transcript.
- If the best hook is not right at the start of the clip candidate, adjust start_time so the hook appears as early as possible.
- The hook must work as an opening on-screen caption to hold viewers in the first 3 seconds.

TYPOGRAPHY PLAN (KINETIC TYPOGRAPHY):
- Choose 3-6 SINGLE words that are the most weighty, emotional, or most worth emphasizing in each clip.
- For each word, set:
  1. 'kata_utama': that specific word, spelled exactly as it appears in the transcript.
  2. 'scale_level': choose 1, 2, or 3.
     - 1 = normal/small
     - 2 = large/emphasis
     - 3 = huge/very crucial
  3. 'style': choose "utama" or "khusus".
  4. 'animasi': choose "bounce_pop" or "stagger_up".
- Do not pick long phrases. Single words only.
- Prioritize words that are strongest in emotion, meaning, or visual retention.

B-ROLL (REQUIRED WHEN RELEVANT):
- Find at most 1-3 moments in the clip that are great for inserting B-roll / stock footage.
- Each B-roll is 3-7 seconds long.
- Provide:
  - start_time
  - end_time
  - search_query
- search_query must be short, clear, and in English.
- Do not place B-roll at the exact same seconds as the hook.
- Only add B-roll if it genuinely helps visualize what is being said.
- If no moment fits, set broll_list to an empty array [].

VISUAL B-ROLL HOOK (FIRST 0-3 SECONDS):
- Provide 2-5 opening B-roll ideas that are contrasting, funny, dramatic, or curiosity-provoking before the original video comes in.
- Include YouTube/TikTok search keywords for the editor.
- If there is a gesture usable as a visual hook, include a reference to it too.
- This is stored in the 'recommended_visual_broll_hook' object and serves only as a reference if the editor wants to manually source footage for the first 3 seconds.

BGM MOOD (BACKGROUND MUSIC):
- Analyze the emotion and topic of this clip.
- Choose ONE background-music mood that fits best from this fixed list: [chill, epic, sad, upbeat, suspense].
- Make sure the mood matches the story. (Example: a tough struggle story = sad/epic, a funny/casual story = chill/upbeat).

SLOW CLOSING:
- end_time MUST be padded by +0.10 to +0.85 seconds after the last word so the ending feels relaxed and is not cut off abruptly.

SELECTION REASONING:
- Fill the 'alasan' field with a brief explanation of why this clip is worth choosing.
- Focus on emotional value, hook strength, retention potential, shareability, and payoff.
- Explain the main viral trigger of this clip.
- Explain why people are likely to watch to the end.
- Explain why this clip stays interesting even without the full video's context.

METADATA LANGUAGE RULES (IMPORTANT):
- ALL generated text MUST be in natural, native English. Do NOT output any non-English text in any field.
- This applies to every text field: title_inggris, hastag, description_hook, description_context, keyword_tags, and tiktok_caption.
- Use natural, concise, readable English suited to short-form content.
- Avoid stiff, word-for-word phrasing.

CROSS-PLATFORM METADATA:
For each clip, produce the following metadata:

1. title_inggris
- Natural, strong, sharp, and easy-to-read English.
- This is the main title used for platform metadata.
- Maximum 100 characters.
- Focus on one main idea.
- Relevant to the clip's content, not the full video in general.
- No cheap clickbait.
- Do not use excessive capitalization.
- Avoid excessive punctuation like !!! ??? ...
- Do not be too generic.

2. hastag
- Provide only 2 to 3 hashtags in a single string.
- All hashtags MUST be in English.
- Separate them with spaces.
- They must be directly relevant to the clip's topic.
- No duplicates.
- Avoid overly generic hashtags like #fyp #viral #trending unless they are truly relevant.
- Use a format like: #mindset #career #productivity

3. description_hook
- Exactly 1 sentence.
- MUST be in English.
- This is the opening sentence of the metadata.
- It must be short, strong, and curiosity-provoking.
- No fake clickbait.

4. description_context
- Exactly 1 sentence.
- MUST be in English.
- Briefly explains the main context of the clip's content.
- Must be relevant to what is said in the clip.

5. keyword_tags
- Contains 5 to 8 short keywords.
- MUST be in English.
- Not hashtags.
- A list of short phrases relevant to the clip's content.
- Avoid keyword spam.
- Prefer keywords people might actually search for.
- This field is primarily for YouTube metadata.

6. tiktok_caption
- 1 to 2 short sentences.
- MUST be in English.
- A more natural, light, and conversational style.
- Stay true to the clip's content.
- Don't just copy-paste the title.
- Don't be too formal.
- Try to keep it under 140 characters.

METADATA QUALITY RULES:
- All metadata must match the clip's content, not the long video in general.
- Do not make promises that aren't discussed in the clip.
- Do not use false hyperbole like "100% guaranteed", "you'll definitely get rich", etc. unless it is very clearly stated.
- If there are numbers, strong phrases, or sharp statements in the original speech, prioritize them as inspiration for the title/caption.
- Title, descriptions, and caption must complement each other, not repeat the same sentence.
- All metadata fields used for platforms must be in natural English, not stiff literal translations.

OUTPUT RULES:
- Output MUST be a valid JSON array.
- Do not add any explanation outside the JSON.
- All fields must be filled.
- When in doubt, prioritize accuracy to the clip's content over excessive creativity.
{_hook_v2_prompt}{_segment_prompt}

REQUIRED JSON STRUCTURE (Follow these field names exactly):
[
  {{
    "rank": 1,
    "viral_score": 95,
    "start_time": 30.5,
    "end_time": 90.0,
    "hook_start_time": 30.5,
    "hook_end_time": 35.0,
    "bgm_mood": "mood_here",
    "typography_plan": [{{ "kata_utama": "...", "scale_level": 2, "style": "utama", "animasi": "bounce_pop" }}],
    "broll_list": [{{ "start_time": 40.0, "end_time": 45.0, "search_query": "..." }}],
    "recommended_visual_broll_hook": [
      {{ "broll_idea": "...", "search_keyword": "...", "why_it_works": "..." }}
    ],
    "hook_v2": {{
      "enabled": true,
      "items": [{{ "start_time": 31.0, "end_time": 32.5, "text": "KEYWORD" }}],
      "transition": {{ "type": "white_flash" }}
    }},
    "keep_segments": [
      {{ "start_time": 30.5, "end_time": 55.0 }},
      {{ "start_time": 58.0, "end_time": 90.0 }}
    ],
    "title_inggris": "...",
    "hastag": "#hashtag1 #hashtag2",
    "description_hook": "...",
    "description_context": "...",
    "keyword_tags": ["tag1", "tag2"],
    "tiktok_caption": "...",
    "alasan": "..."
  }}
]

Transcript:
{transkrip_lengkap}
"""


def _build_clips_schema() -> dict:
    """Strict JSON schema describing the array of clip objects the AI must return.

    Shared by every provider that supports schema-guided JSON (currently NVIDIA's
    ``guided_json``). Returned as a top-level ``array`` schema."""
    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "rank": {"type": "integer"},
                "viral_score": {"type": "integer"},
                "start_time": {"type": "number"},
                "end_time": {"type": "number"},
                "hook_start_time": {"type": "number"},
                "hook_end_time": {"type": "number"},
                "bgm_mood": {
                    "type": "string",
                    "enum": ["chill", "epic", "sad", "upbeat", "suspense"]
                },
                "typography_plan": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "kata_utama": {"type": "string"},
                            "scale_level": {"type": "integer", "enum": [1, 2, 3]},
                            "style": {"type": "string", "enum": ["utama", "khusus"]},
                            "animasi": {"type": "string", "enum": ["bounce_pop", "stagger_up"]}
                        },
                        "required": ["kata_utama", "scale_level", "style", "animasi"]
                    }
                },
                "broll_list": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "start_time": {"type": "number"},
                            "end_time": {"type": "number"},
                            "search_query": {"type": "string"}
                        },
                        "required": ["start_time", "end_time", "search_query"]
                    }
                },
                "recommended_visual_broll_hook": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "broll_idea": {"type": "string"},
                            "search_keyword": {"type": "string"},
                            "why_it_works": {"type": "string"}
                        },
                        "required": ["broll_idea", "search_keyword", "why_it_works"]
                    }
                },
                "title_inggris": {"type": "string"},
                "hastag": {"type": "string"},
                "description_hook": {"type": "string"},
                "description_context": {"type": "string"},
                "keyword_tags": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "tiktok_caption": {"type": "string"},
                "alasan": {"type": "string"},
                "hook_v2": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "start_time": {"type": "number"},
                                    "end_time": {"type": "number"},
                                    "text": {"type": "string"},
                                },
                                "required": ["start_time", "end_time", "text"],
                            },
                        },
                        "transition": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "type": {"type": "string", "enum": ["white_flash", "glitch"]},
                            },
                            "required": ["type"],
                        },
                    },
                    "required": ["enabled", "items", "transition"],
                },
                "keep_segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "start_time": {"type": "number"},
                            "end_time": {"type": "number"},
                        },
                        "required": ["start_time", "end_time"],
                    },
                },
            },
            "required": [
                "rank", "viral_score", "start_time", "end_time", "hook_start_time", "hook_end_time",
                "bgm_mood", "typography_plan", "broll_list", "recommended_visual_broll_hook",
                "title_inggris", "hastag", "description_hook", "description_context",
                "keyword_tags", "tiktok_caption",
                "alasan", "hook_v2", "keep_segments"
            ]
        }
    }

def _coerce_clip_list(content: str, provider: str) -> list[dict]:
    """Parse a model's text response into a list of clip dicts.

    Strips markdown code fences, ``json.loads``-es, and unwraps a top-level
    object that wraps the array under a common key. Shared by all
    non-Gemini providers."""
    if not content or not content.strip():
        raise ValueError(f"Provider {provider} returned an empty response.")

    content = content.strip()
    if "```" in content:
        content = re.sub(r"```(json)?", "", content).strip()
        content = content.split("```")[0].strip()

    hasil = json.loads(content)

    if isinstance(hasil, dict):
        for key in ("clips", "data", "highlights", "results"):
            if key in hasil and isinstance(hasil[key], list):
                return hasil[key]
        return [hasil]

    if not isinstance(hasil, list):
        raise ValueError(f"Provider {provider} returned non-list/dict format: {type(hasil)}")

    return hasil


def analyze_with_nvidia(transkrip_lengkap: str, cfg) -> list[dict]:
    """Analyze transcript using NVIDIA NIM API (OpenAI compatible, guided JSON)."""
    from openai import OpenAI

    print(f"[3/3] Analyzing top {cfg.jumlah_clip} moments using NVIDIA ({cfg.nvidia_model})...")

    if not cfg.api_key_nvidia:
        raise ValueError("NVIDIA_API_KEY not found in environment.")

    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=cfg.api_key_nvidia)
    prompt = get_analysis_prompt(transkrip_lengkap, cfg.jumlah_clip, cfg.durasi_hook, cfg=cfg)

    completion = client.chat.completions.create(
        model=cfg.nvidia_model,
        messages=[
            {"role": "system", "content": "You are a professional video editor and strategist. Return JSON only. Follow the provided JSON schema exactly."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        top_p=1,
        max_tokens=16384,
        extra_body={
            "chat_template_kwargs": {"thinking": False},
            "nvext": {"guided_json": _build_clips_schema()},
        },
    )
    return _coerce_clip_list(completion.choices[0].message.content, "NVIDIA")


def analyze_with_openai(transkrip_lengkap: str, cfg) -> list[dict]:
    """Analyze transcript using the OpenAI API (JSON object mode)."""
    from openai import OpenAI

    model = getattr(cfg, "openai_model", "gpt-4o")
    print(f"[3/3] Analyzing top {cfg.jumlah_clip} moments using OpenAI ({model})...")

    if not getattr(cfg, "api_key_openai", ""):
        raise ValueError("OPENAI_API_KEY not found in environment.")

    client = OpenAI(api_key=cfg.api_key_openai)
    prompt = get_analysis_prompt(transkrip_lengkap, cfg.jumlah_clip, cfg.durasi_hook, cfg=cfg)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional video editor and viral content strategist. "
                    "Respond with a single JSON object of the form {\"clips\": [ ... ]} whose "
                    "items follow the schema described in the user's instructions exactly. "
                    "Output JSON only - no markdown, no prose."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=16000,
        response_format={"type": "json_object"},
    )
    return _coerce_clip_list(completion.choices[0].message.content, "OpenAI")


def analyze_with_anthropic(transkrip_lengkap: str, cfg) -> list[dict]:
    """Analyze transcript using the Anthropic Claude API.

    Claude 4.x rejects assistant prefills and sampling params, so JSON is forced
    via a strict system prompt and parsed from the text response. Streamed so a
    large clip array can't hit the SDK's non-streaming timeout guard."""
    import anthropic

    model = getattr(cfg, "anthropic_model", "claude-opus-4-8")
    print(f"[3/3] Analyzing top {cfg.jumlah_clip} moments using Anthropic ({model})...")

    if not getattr(cfg, "api_key_anthropic", ""):
        raise ValueError("ANTHROPIC_API_KEY not found in environment.")

    client = anthropic.Anthropic(api_key=cfg.api_key_anthropic)
    prompt = get_analysis_prompt(transkrip_lengkap, cfg.jumlah_clip, cfg.durasi_hook, cfg=cfg)
    system = (
        "You are a professional video editor and viral content strategist. "
        "Return ONLY a JSON array of clip objects that conforms exactly to the schema "
        "described in the user's instructions. Do not include markdown, code fences, or "
        "any prose - your entire response must start with '[' and end with ']'."
    )

    with client.messages.stream(
        model=model,
        max_tokens=32000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        message = stream.get_final_message()

    if message.stop_reason == "refusal":
        raise ValueError("Anthropic declined the request (safety refusal).")

    text = "".join(block.text for block in message.content if block.type == "text")
    return _coerce_clip_list(text, "Anthropic")


def analyze_with_ai(transkrip_lengkap: str, cfg) -> list[dict]:
    """Dispatcher for AI analysis based on provider, with Gemini as fallback."""
    provider = getattr(cfg, "ai_provider", "gemini")

    providers = {
        "nvidia": (getattr(cfg, "api_key_nvidia", ""), "NVIDIA_API_KEY", analyze_with_nvidia),
        "openai": (getattr(cfg, "api_key_openai", ""), "OPENAI_API_KEY", analyze_with_openai),
        "anthropic": (getattr(cfg, "api_key_anthropic", ""), "ANTHROPIC_API_KEY", analyze_with_anthropic),
    }

    if provider in providers:
        api_key, env_name, analyze_fn = providers[provider]
        if not api_key:
            print(f"⚠️ {env_name} not found! Falling back to Gemini...")
        else:
            try:
                return analyze_fn(transkrip_lengkap, cfg)
            except Exception as e:
                print(f"⚠️ {provider} API failed: {e}. Falling back to Gemini...")

    return analyze_with_gemini(transkrip_lengkap, cfg)


def analyze_with_gemini(
    transkrip_lengkap: str,
    cfg,
) -> list[dict]:
    """Analyse transcript with Gemini AI."""
    import google.genai as genai
    from google.genai import types

    print(f"[3/3] Analyzing top {cfg.jumlah_clip} best moments using Gemini...")

    prompt = get_analysis_prompt(transkrip_lengkap, cfg.jumlah_clip, cfg.durasi_hook, cfg=cfg)

    # JSON Schema definitions (same as before)
    schema_broll = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "start_time": {"type": "NUMBER"},
                "end_time": {"type": "NUMBER"},
                "search_query": {"type": "STRING"},
            },
            "required": ["start_time", "end_time", "search_query"],
        },
    }

    schema_visual_broll_hook = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "broll_idea": {"type": "STRING"},
                "search_keyword": {"type": "STRING"},
                "why_it_works": {"type": "STRING"},
            },
            "required": ["broll_idea", "search_keyword", "why_it_works"],
        },
    }

    schema_typography = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "kata_utama": {"type": "STRING"},
                "scale_level": {"type": "INTEGER"},
                "style": {"type": "STRING"},
                "animasi": {"type": "STRING"},
            },
            "required": ["kata_utama", "scale_level", "style", "animasi"],
        },
    }

    client = genai.Client(
        api_key=cfg.api_key_gemini,
        http_options=types.HttpOptions(
            timeout=REQUEST_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )

    gemini_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema={
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "rank": {"type": "INTEGER"},
                    "viral_score": {"type": "INTEGER"},
                    "hook_start_time": {"type": "NUMBER"},
                    "hook_end_time": {"type": "NUMBER"},
                    "start_time": {"type": "NUMBER"},
                    "end_time": {"type": "NUMBER"},
                    "typography_plan": schema_typography,
                    "broll_list": schema_broll,
                    "recommended_visual_broll_hook": schema_visual_broll_hook,
                    "alasan": {"type": "STRING"},
                    "bgm_mood": {"type": "STRING"},
                    "title_inggris": {"type": "STRING"},
                    "hastag": {"type": "STRING"},
                    "description_hook": {"type": "STRING"},
                    "description_context": {"type": "STRING"},
                    "keyword_tags": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "tiktok_caption": {"type": "STRING"},
                    "hook_v2": {
                        "type": "OBJECT",
                        "properties": {
                            "enabled": {"type": "BOOLEAN"},
                            "items": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "start_time": {"type": "NUMBER"},
                                        "end_time": {"type": "NUMBER"},
                                        "text": {"type": "STRING"},
                                    },
                                    "required": ["start_time", "end_time", "text"],
                                },
                            },
                            "transition": {
                                "type": "OBJECT",
                                "properties": {
                                    "type": {"type": "STRING"},
                                },
                                "required": ["type"],
                            },
                        },
                        "required": ["enabled", "items", "transition"],
                    },
                    "keep_segments": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "start_time": {"type": "NUMBER"},
                                "end_time": {"type": "NUMBER"},
                            },
                            "required": ["start_time", "end_time"],
                        },
                    },
                },
                "required": [
                    "rank", "viral_score", "hook_start_time", "hook_end_time",
                    "start_time", "end_time", "typography_plan",
                    "broll_list", "recommended_visual_broll_hook", "alasan", "bgm_mood",
                    "title_inggris", "hastag",
                    "description_hook", "description_context",
                    "keyword_tags", "tiktok_caption",
                    "hook_v2", "keep_segments",
                ],
            },
        },
    )

    return _generate_json_with_retry(
        client=client,
        model=cfg.gemini_model,
        fallback_model=getattr(cfg, "gemini_fallback_model", None),
        contents=prompt,
        config=gemini_config,
    )