"""
clipping.engine — Download, Transcription & Gemini AI Analysis

Maps to Cell 2 (The Engine) of the notebook.
"""

import json
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


def download_video(
    url: str,
    output_path: str,
    use_dlp_subs: bool = False,
    download_source_height: str | int = "max",
) -> None:
    """
    Download a YouTube video to *output_path* with configurable source height.
    """
    print("[1/3] Mendownload video dari YouTube...")
    if download_source_height == "max":
        print("      🎯 Source quality: highest available", flush=True)
    else:
        print(f"      🎯 Source quality: up to {download_source_height}p", flush=True)

    ydl_opts = {
        "format": _build_ydl_format_selector(download_source_height),
        "outtmpl": output_path,
        "quiet": True,
        "merge_output_format": "mp4",
        "remote_components": ["ejs:github"],
    }

    if use_dlp_subs:
        print("      Mencoba mencari subtitle bahasa otomatis (en / id)...")
        import glob

        for lang in ["en", "id"]:
            ydl_opts_subs = ydl_opts.copy()
            ydl_opts_subs.update({
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": [lang],
                "subtitlesformat": "json3",
                "skip_download": True,  # Hanya fokus download subtitle
            })

            try:
                with YoutubeDL(ydl_opts_subs) as ydl:
                    ydl.download([url])

                # Cek apakah json3 untuk bahasa ini benar-benar terdownload
                if glob.glob(output_path.replace(".mp4", f".*.json3")):
                    print(f"      ✅ Subtitle '{lang}' ditemukan. Melanjutkan ke video...")
                    break
            except Exception as e:
                print(f"      ⚠️ Gagal menarik subtitle '{lang}' ({e}). Mencoba opsi selanjutnya...")

    # Jalankan download video terpisah dari urusan subtitle
    with YoutubeDL(ydl_opts) as ydl:
        # Extra step to verify resolution before downloading
        try:
            info = ydl.extract_info(url, download=False)
            best_h = info.get("height", "unknown")
            v_codec = info.get("vcodec", "unknown")
            print(f"      ✅ Mendownload: {best_h}p (Codec: {v_codec})", flush=True)
        except Exception as e:
            print(f"      ⚠️ Gagal mengecek info detail: {e}", flush=True)

        ydl.download([url])


# ==============================================================================
# TAHAP 2: TRANSKRIPSI WHISPER & JSON3 FALLBACK
# ==============================================================================

def parse_youtube_json3_subs(json_path: str, max_words_per_subtitle: int = 5) -> tuple[str, list[dict]]:
    """
    Parse downloaded YouTube JSON3 subtitles into transkrip_lengkap and data_segmen.
    Returns empty string/list if parsing fails.
    """
    import json

    print("[2/3] Memproses subtitle JSON3 dari YouTube...")
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
        print(f"⚠️ Gagal memparsing JSON3: {e}")
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
    print("[2/3] Memulai transkripsi dengan Faster-Whisper (Level Per-Kata)...")

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
# TAHAP 3: ANALISIS GEMINI AI
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
                raise ValueError("Gemini mengembalikan response.text kosong.")

            return json.loads(text)

        except Exception as exc:
            last_exc = exc
            status_code = _extract_status_code(exc)
            retryable = _is_retryable(exc)

            print(
                f"[Gemini] Attempt {attempt}/{MAX_ATTEMPTS} gagal | "
                f"status={status_code} | error={exc}"
            )

            if (not retryable) or attempt == MAX_ATTEMPTS:
                break

            wait_seconds = INITIAL_WAIT_SECONDS + ((attempt - 1) * WAIT_INCREMENT_SECONDS)
            print(f"[Gemini] Retry lagi dalam {wait_seconds} detik...")
            time.sleep(wait_seconds)

    print(f"[Gemini] Percobaan dengan model utama ({model}) gagal.")
    if fallback_model:
        print(f"[Gemini] Mencoba satu kali lagi dengan fallback model ({fallback_model})...")
        try:
            response = client.models.generate_content(
                model=fallback_model,
                contents=contents,
                config=config,
            )
            text = getattr(response, "text", None)
            if not text or not text.strip():
                raise ValueError("Gemini fallback mengembalikan response.text kosong.")

            return json.loads(text)
        except Exception as exc_fallback:
            print(f"[Gemini] Fallback model gagal | error={exc_fallback}")
            raise RuntimeError(
                f"Gagal memanggil Gemini utama & fallback. "
                f"Laporan Utama status={status_code}, error={last_exc} | "
                f"Laporan Fallback error={exc_fallback}"
            ) from exc_fallback

    raise RuntimeError(
        f"Gagal memanggil Gemini setelah {MAX_ATTEMPTS} percobaan. Error terakhir: {last_exc}"
    ) from last_exc


def get_analysis_prompt(transkrip_lengkap: str, jumlah_clip: int, durasi_hook: int) -> str:
    """Centralized prompt for both Gemini and NVIDIA providers."""
    return f"""
Kamu adalah Art Director, Editor Video, dan Strategist Metadata Short-Form Content untuk TikTok, Reels, dan YouTube Shorts.

Baca transkrip video berikut. Format transkrip:
[detik_mulai - detik_selesai] teks

TUGAS UTAMA:
- Carikan {jumlah_clip} momen paling menarik, paling kuat, paling shareable, dan paling berpotensi viral untuk dijadikan klip pendek.
- Urutkan dari Peringkat 1 (paling bagus) sampai {jumlah_clip}.
- Untuk setiap klip, hasilkan timing klip, hook, typography plan, b-roll plan, alasan pemilihan, dan metadata lintas platform.
- Semua output harus sangat relevan dengan isi klip, bukan isi video penuh secara umum.

ATURAN PEMILIHAN KLIP:
- Durasi klip harus 30-180 detik.
- Pilih bagian yang punya emosi, konflik, kejutan, insight, opini kuat, pelajaran praktis, atau punchline jelas.
- Utamakan bagian yang tetap menarik walau ditonton tanpa konteks video penuh.
- Hindari klip yang isinya terlalu mirip satu sama lain.
- Jangan pilih klip yang terasa datar, bertele-tele, atau tidak punya payoff yang jelas.

HOOK (WAJIB):
- Ambil 1 kalimat paling punchy yang ADA DI DALAM klip.
- Hook harus terasa kuat dan menarik perhatian dalam ~{durasi_hook} detik pertama.
- Simpan sebagai hook_start_time dan hook_end_time.
- Hook harus membuat orang ingin lanjut menonton, tapi jangan clickbait palsu.
- Pastikan hook masih natural dan benar-benar diucapkan dalam transkrip.

TYPOGRAPHY PLAN (KINETIC TYPOGRAPHY):
- Pilih 3-6 kata TUNGGAL paling berbobot, emosional, atau paling layak ditekankan dari setiap klip.
- Untuk setiap kata, tentukan:
  1. 'kata_utama': kata spesifik tersebut, harus sama persis ejaannya dengan transkrip.
  2. 'scale_level': pilih 1, 2, atau 3.
     - 1 = normal/kecil
     - 2 = besar/penekanan
     - 3 = raksasa/sangat krusial
  3. 'style': pilih "utama" atau "khusus".
  4. 'animasi': pilih "bounce_pop" atau "stagger_up".
- Jangan pilih frasa panjang. Hanya kata tunggal.
- Prioritaskan kata yang paling kuat secara emosi, makna, atau retensi visual.

B-ROLL (WAJIB JIKA RELEVAN):
- Carikan maksimal 1-3 momen dalam klip yang sangat cocok disisipi video B-roll / stock footage.
- Setiap B-roll berdurasi 3-7 detik.
- Berikan:
  - start_time
  - end_time
  - search_query
- search_query harus singkat, jelas, dan dalam Bahasa Inggris.
- Jangan taruh B-roll tepat di detik yang sama dengan hook.
- Hanya tambahkan B-roll jika benar-benar membantu visualisasi isi ucapan.
- Jika tidak ada momen yang cocok, isi broll_list dengan array kosong [].

BGM MOOD (BACKGROUND MUSIC):
- Analisis emosi dan topik dari klip ini.
- Pilih SATU mood musik latar yang paling cocok dari daftar baku ini: [chill, epic, sad, upbeat, suspense].
- Pastikan mood selaras dengan cerita. (Contoh: cerita perjuangan berat = sad/epic, cerita lucu/santai = chill/upbeat).

SLOW CLOSING:
- end_time HARUS ditambah padding +0.10 sampai +0.85 detik setelah kata terakhir agar ending terasa lega dan tidak kepotong kasar.

ALASAN PEMILIHAN:
- Isi field 'alasan' dengan penjelasan singkat mengapa klip ini layak dipilih.
- Fokus pada nilai emosi, kekuatan hook, potensi retention, dan shareability.

ATURAN BAHASA METADATA:
- title_indonesia tetap wajib diisi untuk kompatibilitas internal / fallback.
- title_indonesia HARUS dalam Bahasa Indonesia natural dan maksimal 100 karakter.
- Semua metadata lintas platform utama harus berbahasa Inggris natural.
- Ini berlaku untuk:
  - title_inggris
  - hastag
  - description_hook
  - description_context
  - keyword_tags
  - tiktok_caption
- Khusus kebutuhan TikTok versi Indonesia, juga buat:
  - tiktok_title_id
  - tiktok_caption_id
- tiktok_title_id dan tiktok_caption_id HARUS dalam Bahasa Indonesia natural.
- tiktok_title_id harus lebih deskriptif daripada title_indonesia, boleh lebih panjang dari 100 karakter jika perlu, dan harus menjelaskan isi klip/video dengan jelas.
- tiktok_caption_id harus natural, informatif, cocok untuk audiens Indonesia, dan boleh sedikit lebih panjang jika itu membantu menjelaskan isi klip.
- Jangan mencampur Bahasa Indonesia dan Bahasa Inggris di dalam field yang sama.
- Gunakan English yang natural, ringkas, enak dibaca, dan cocok untuk short-form content.
- Hindari terjemahan literal yang kaku.

METADATA LINTAS PLATFORM:
Untuk setiap klip, hasilkan metadata berikut:

1. title_indonesia
- Bahasa Indonesia natural, singkat, dan relevan.
- Ini hanya untuk kompatibilitas internal / fallback.
- Maksimal 100 karakter.

2. title_inggris
- Bahasa Inggris natural, kuat, tajam, dan enak dibaca.
- Ini adalah judul utama untuk metadata platform.
- Maksimal 100 karakter.
- Fokus pada 1 ide utama.
- Relevan dengan isi klip, bukan isi video penuh secara umum.
- Jangan clickbait murahan.
- Jangan pakai huruf kapital berlebihan.
- Hindari tanda baca berlebihan seperti !!! ??? ...
- Jangan terlalu generik.

3. hastag
- Isi dengan 2 sampai 3 hashtag saja dalam satu string.
- Semua hashtag HARUS dalam Bahasa Inggris.
- Pisahkan dengan spasi.
- Harus relevan langsung dengan topik klip.
- Jangan duplikat.
- Hindari hashtag terlalu generik seperti #fyp #viral #trending kecuali memang sangat relevan.
- Gunakan format seperti: #mindset #career #productivity

4. description_hook
- Tepat 1 kalimat.
- HARUS dalam Bahasa Inggris.
- Ini adalah kalimat pembuka metadata.
- Harus singkat, kuat, dan memancing rasa ingin tahu.
- Jangan clickbait palsu.

5. description_context
- Tepat 1 kalimat.
- HARUS dalam Bahasa Inggris.
- Menjelaskan konteks utama isi klip secara ringkas.
- Harus relevan dengan pembicaraan di klip.

6. keyword_tags
- Berisi 5 sampai 8 keyword pendek.
- HARUS dalam Bahasa Inggris.
- Bukan hashtag.
- Harus berupa daftar frasa singkat yang relevan dengan isi klip.
- Hindari keyword spam.
- Utamakan keyword yang mungkin benar-benar dicari orang.
- Field ini terutama untuk kebutuhan metadata YouTube.

7. tiktok_title_id
- Bahasa Indonesia natural.
- Lebih panjang dan lebih menjelaskan isi video daripada title_indonesia.
- Tidak perlu dibatasi 100 karakter, tapi tetap harus ringkas, jelas, dan enak dibaca.
- Harus relevan dengan isi klip, bukan isi video panjang secara umum.
- Jangan clickbait murahan.

8. tiktok_caption_id
- 1 sampai 2 kalimat.
- HARUS dalam Bahasa Indonesia.
- Boleh sedikit lebih panjang daripada caption English jika membantu menjelaskan isi klip.
- Gaya natural, ringan, dan enak dibaca.
- Tetap sesuai isi klip.
- Jangan sekadar copy-paste title.
- Jangan terlalu formal.

9. tiktok_caption
- 1 sampai 2 kalimat singkat.
- HARUS dalam Bahasa Inggris.
- Gaya lebih natural, ringan, dan conversational.
- Tetap sesuai isi klip.
- Jangan sekadar copy-paste title.
- Jangan terlalu formal.
- Usahakan tidak lebih dari 140 karakter.

ATURAN KUALITAS METADATA:
- Semua metadata harus sesuai isi klip, bukan isi video panjang secara umum.
- Jangan membuat janji yang tidak dibahas di klip.
- Jangan pakai hiperbola palsu seperti "100% berhasil", "pasti kaya", dll kecuali memang sangat jelas disebutkan.
- Jika ada angka, frasa kuat, atau statement tajam dari ucapan asli, prioritaskan itu sebagai inspirasi judul/caption.
- Title, descriptions, dan caption harus saling melengkapi, bukan mengulang kalimat yang sama.
- Semua field metadata yang dipakai untuk platform harus berbahasa Inggris natural, bukan terjemahan literal yang kaku.
- Khusus tiktok_title_id dan tiktok_caption_id, gunakan Bahasa Indonesia yang natural, jelas, dan lebih menjelaskan isi klip untuk audiens Indonesia.

ATURAN OUTPUT:
- Output HARUS berupa JSON array valid.
- Jangan beri penjelasan apa pun di luar JSON.
- Semua field wajib terisi.
- Jika ragu, prioritaskan akurasi isi klip daripada kreativitas berlebihan.

STRUKTUR JSON WAJIB (Ikuti nama field ini secara kaku):
[
  {{
    "rank": 1,
    "start_time": 30.5,
    "end_time": 90.0,
    "hook_start_time": 30.5,
    "hook_end_time": 35.0,
    "bgm_mood": "mood_here",
    "typography_plan": [{{ "kata_utama": "...", "scale_level": 2, "style": "utama", "animasi": "bounce_pop" }}],
    "broll_list": [{{ "start_time": 40.0, "end_time": 45.0, "search_query": "..." }}],
    "title_indonesia": "...",
    "title_inggris": "...",
    "hastag": "#hastag1 #hastag2",
    "description_hook": "...",
    "description_context": "...",
    "keyword_tags": ["tag1", "tag2"],
    "tiktok_title_id": "...",
    "tiktok_caption_id": "...",
    "tiktok_caption": "...",
    "alasan": "..."
  }}
]

Transkrip:
{transkrip_lengkap}
"""


def analyze_with_nvidia(transkrip_lengkap: str, cfg) -> list[dict]:
    """Analyze transcript using NVIDIA NIM API (OpenAI compatible)."""
    from openai import OpenAI
    
    print(f"[3/3] Menganalisis Top {cfg.jumlah_clip} momen menggunakan NVIDIA ({cfg.nvidia_model})...")
    
    if not cfg.api_key_nvidia:
        raise ValueError("NVIDIA_API_KEY tidak ditemukan di environment.")

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=cfg.api_key_nvidia
    )
    
    prompt = get_analysis_prompt(transkrip_lengkap, cfg.jumlah_clip, cfg.durasi_hook)
    
    # Define the strict schema for Guided JSON (NVIDIA NIM specific)
    clips_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "rank": {"type": "integer"},
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
                "title_indonesia": {"type": "string"},
                "title_inggris": {"type": "string"},
                "hastag": {"type": "string"},
                "description_hook": {"type": "string"},
                "description_context": {"type": "string"},
                "keyword_tags": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "tiktok_title_id": {"type": "string"},
                "tiktok_caption_id": {"type": "string"},
                "tiktok_caption": {"type": "string"},
                "alasan": {"type": "string"}
            },
            "required": [
                "rank", "start_time", "end_time", "hook_start_time", "hook_end_time",
                "bgm_mood", "typography_plan", "broll_list", "title_indonesia",
                "title_inggris", "hastag", "description_hook", "description_context",
                "keyword_tags", "tiktok_title_id", "tiktok_caption_id", "tiktok_caption",
                "alasan"
            ]
        }
    }

    completion = client.chat.completions.create(
        model=cfg.nvidia_model,
        messages=[
            {"role": "system", "content": "You are a professional video editor and strategist. Return JSON only. Follow the provided JSON schema exactly."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        top_p=1,
        max_tokens=16384,
        extra_body={
            "chat_template_kwargs": {"thinking": False},
            "nvext": {
                "guided_json": clips_schema
            }
        }
    )
    
    content = completion.choices[0].message.content
    
    if "```" in content:
        content = re.sub(r"```(json)?", "", content).strip()
        content = content.split("```")[0].strip()
        
    hasil = json.loads(content)
    
    # Guided JSON should return an array directly if schema says type: array
    # but we keep the unwrapper just in case of non-conforming fallbacks
    if isinstance(hasil, dict):
        for key in ["clips", "data", "highlights"]:
            if key in hasil and isinstance(hasil[key], list):
                hasil = hasil[key]
                break
                
    if not isinstance(hasil, list):
        if isinstance(hasil, dict):
            return [hasil]
        raise ValueError(f"Provider NVIDIA mengembalikan format non-list/dict: {type(hasil)}")
        
    return hasil


def analyze_with_ai(transkrip_lengkap: str, cfg) -> list[dict]:
    """Dispatcher for AI analysis based on provider."""
    provider = getattr(cfg, "ai_provider", "gemini")
    
    if provider == "nvidia":
        if not cfg.api_key_nvidia:
            print("⚠️ NVIDIA_API_KEY tidak ditemukan! Mencoba fallback ke Gemini...")
        else:
            try:
                return analyze_with_nvidia(transkrip_lengkap, cfg)
            except Exception as e:
                print(f"⚠️ NVIDIA API gagal: {e}. Fallback ke Gemini...")
    
    return analyze_with_gemini(transkrip_lengkap, cfg)


def analyze_with_gemini(
    transkrip_lengkap: str,
    cfg,
) -> list[dict]:
    """Analyse transcript with Gemini AI."""
    import google.genai as genai
    from google.genai import types

    print(f"[3/3] Menganalisis Top {cfg.jumlah_clip} momen terbaik menggunakan Gemini...")

    prompt = get_analysis_prompt(transkrip_lengkap, cfg.jumlah_clip, cfg.durasi_hook)

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
                    "hook_start_time": {"type": "NUMBER"},
                    "hook_end_time": {"type": "NUMBER"},
                    "start_time": {"type": "NUMBER"},
                    "end_time": {"type": "NUMBER"},
                    "typography_plan": schema_typography,
                    "broll_list": schema_broll,
                    "alasan": {"type": "STRING"},
                    "bgm_mood": {"type": "STRING"},
                    "title_indonesia": {"type": "STRING"},
                    "title_inggris": {"type": "STRING"},
                    "hastag": {"type": "STRING"},
                    "description_hook": {"type": "STRING"},
                    "description_context": {"type": "STRING"},
                    "keyword_tags": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "tiktok_title_id": {"type": "STRING"},
                    "tiktok_caption_id": {"type": "STRING"},
                    "tiktok_caption": {"type": "STRING"},
                },
                "required": [
                    "rank", "hook_start_time", "hook_end_time",
                    "start_time", "end_time", "typography_plan",
                    "broll_list", "alasan", "bgm_mood",
                    "title_indonesia", "title_inggris", "hastag",
                    "description_hook", "description_context",
                    "keyword_tags", "tiktok_title_id",
                    "tiktok_caption_id", "tiktok_caption",
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


    return hasil_json