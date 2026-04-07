"""
clipping.diarization — Speaker Diarization with Pyannote

Deteksi speaker berbeda dalam audio/video menggunakan model
pyannote/speaker-diarization-3.1. Digunakan untuk fitur split-screen
pada klip podcast.
"""

import os
import subprocess

# ==============================================================================
# AUDIO EXTRACTION
# ==============================================================================


def extract_audio(video_path: str, audio_output_path: str) -> str:
    """
    Extract audio from video to WAV using FFmpeg.

    Parameters
    ----------
    video_path : str
        Path to input video file.
    audio_output_path : str
        Path for the output WAV file.

    Returns
    -------
    str
        Path to the extracted audio file.
    """
    if os.path.exists(audio_output_path):
        os.remove(audio_output_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",  # no video
        "-acodec",
        "pcm_s16le",  # 16-bit PCM
        "-ar",
        "16000",  # 16kHz sample rate (optimal for Pyannote)
        "-ac",
        "1",  # mono
        audio_output_path,
    ]

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return audio_output_path


# ==============================================================================
# SPEAKER DIARIZATION
# ==============================================================================


def run_diarization(
    audio_path: str,
    hf_token: str,
    num_speakers: int = 2,
) -> list[dict]:
    """
    Run speaker diarization using Pyannote.

    Parameters
    ----------
    audio_path : str
        Path to WAV audio file.
    hf_token : str
        HuggingFace token (required for pyannote model access).
    num_speakers : int
        Expected number of speakers.

    Returns
    -------
    list[dict]
        List of dicts with keys: speaker, start, end.
        Example: [{"speaker": "SPEAKER_00", "start": 0.5, "end": 3.2}, ...]

    Raises
    ------
    RuntimeError
        If diarization fails or produces no segments.
    """
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN tidak ditemukan. Pyannote membutuhkan HuggingFace token. "
            "Set via: export HF_TOKEN='your-token' atau di file .env"
        )

    print("🎙️ Memuat model Pyannote speaker-diarization-3.1...")

    from pyannote.audio import Pipeline

    # Newer pyannote uses 'token', older uses 'use_auth_token'
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
    except TypeError:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )

    # Use GPU if available
    try:
        import torch

        if torch.cuda.is_available():
            import torch

            pipeline.to(torch.device("cuda"))
            print("   ✅ Pyannote menggunakan GPU (CUDA)")
        else:
            print("   ℹ️ Pyannote menggunakan CPU")
    except Exception:
        print("   ℹ️ Pyannote menggunakan CPU")

    print("🎙️ Menjalankan speaker diarization...")
    diarization = pipeline(audio_path, num_speakers=num_speakers)

    # Recent pyannote (3.1+) returns a DiarizeOutput dataclass or similar
    if not hasattr(diarization, "itertracks"):
        if hasattr(diarization, "speaker_diarization"):
            diarization = diarization.speaker_diarization
        elif hasattr(diarization, "annotation"):
            diarization = diarization.annotation
        elif isinstance(diarization, dict) and "annotation" in diarization:
            diarization = diarization["annotation"]
        else:
            # Last resort: log what we got
            print(f"   ⚠️ Diarization output type: {type(diarization)}")

    # Convert to list of dicts
    raw_segments = []
    try:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            raw_segments.append(
                {
                    "speaker": speaker,
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                }
            )
    except AttributeError as e:
        raise RuntimeError(
            f"Gagal memproses hasil diarization ({type(diarization)}): {e}. "
            "Pastikan model pyannote/speaker-diarization-3.1 terinstal dengan benar."
        )

    if not raw_segments:
        raise RuntimeError("Diarization tidak menghasilkan segment apapun.")

    # Merge adjacent segments from same speaker (gap < 0.5s)
    merged = _merge_adjacent_segments(raw_segments, max_gap=0.5)

    # Get unique speakers
    speakers = sorted(set(s["speaker"] for s in merged))
    print(
        f"   ✅ Diarization selesai: {len(merged)} segments, {len(speakers)} speakers ({', '.join(speakers)})"
    )

    return merged


def _merge_adjacent_segments(segments: list[dict], max_gap: float = 0.5) -> list[dict]:
    """Merge consecutive segments from the same speaker if gap < max_gap."""
    if not segments:
        return []

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        prev = merged[-1]
        if seg["speaker"] == prev["speaker"] and (seg["start"] - prev["end"]) < max_gap:
            prev["end"] = seg["end"]
        else:
            merged.append(seg.copy())

    return merged


# ==============================================================================
# HELPER: ACTIVE SPEAKER LOOKUP
# ==============================================================================


def get_active_speaker(
    diarization_data: list[dict],
    timestamp: float,
) -> str | None:
    """
    Determine which speaker is active at a given timestamp.

    Parameters
    ----------
    diarization_data : list[dict]
        Diarization segments from run_diarization().
    timestamp : float
        Time in seconds to query.

    Returns
    -------
    str or None
        Speaker label (e.g. "SPEAKER_00") or None if no speaker active.
    """
    for seg in diarization_data:
        if seg["start"] <= timestamp <= seg["end"]:
            return seg["speaker"]
    return None


def get_active_speakers(
    diarization_data: list[dict],
    timestamp: float,
) -> list[str]:
    """
    Return ALL speakers active at a given timestamp.

    Unlike get_active_speaker(), this returns a list and may include
    more than one speaker when segments overlap (simultaneous speech).

    Parameters
    ----------
    diarization_data : list[dict]
        Diarization segments from run_diarization().
    timestamp : float
        Time in seconds to query.

    Returns
    -------
    list[str]
        List of active speaker labels (may be empty or contain 2+ speakers).
    """
    active = []
    for seg in diarization_data:
        if seg["start"] <= timestamp <= seg["end"]:
            if seg["speaker"] not in active:
                active.append(seg["speaker"])
    return active
