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

def download_video(url: str, output_path: str) -> None:
    """Download a YouTube video to *output_path* (max 1080p)."""
    print("[1/3] Mendownload video dari YouTube...")

    ydl_opts = {
        "format": (
            "best[height<=?1080][ext=mp4][acodec!=none][vcodec!=none]/"
            "bestvideo[height<=?1080][ext=mp4]+bestaudio[ext=m4a]/"
            "best[height<=?1080][acodec!=none][vcodec!=none]/"
            "best"
        ),
        "outtmpl": output_path,
        "quiet": True,
        "merge_output_format": "mp4",
        "remote_components": ["ejs:github"],
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


# ==============================================================================
# TAHAP 2: TRANSKRIPSI WHISPER
# ==============================================================================

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
MAX_ATTEMPTS           = 10
INITIAL_WAIT_SECONDS   = 60
WAIT_INCREMENT_SECONDS = 30
REQUEST_TIMEOUT_MS     = 15 * 60 * 1000  # 15 menit
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


def _generate_json_with_retry(client, model, contents, config):
    last_exc = None

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
                raise RuntimeError(
                    f"Gagal memanggil Gemini setelah {attempt} percobaan. "
                    f"status={status_code}, error={exc}"
                ) from exc

            wait_seconds = INITIAL_WAIT_SECONDS + ((attempt - 1) * WAIT_INCREMENT_SECONDS)
            print(f"[Gemini] Retry lagi dalam {wait_seconds} detik...")
            time.sleep(wait_seconds)

    raise RuntimeError(f"Gagal memanggil Gemini. Error terakhir: {last_exc}") from last_exc


def analyze_with_gemini(
    transkrip_lengkap: str,
    cfg,
) -> list[dict]:
    """
    Analyse transcript with Gemini AI to pick best clip moments.

    Parameters
    ----------
    transkrip_lengkap : str
        The full transcript text.
    cfg : SimpleNamespace
        Config object (must have api_key_gemini, jumlah_clip, durasi_hook, gemini_model).

    Returns
    -------
    list[dict]
        List of clip dicts parsed from Gemini JSON response.
    """
    import google.genai as genai
    from google.genai import types

    jumlah_clip = cfg.jumlah_clip
    durasi_hook = cfg.durasi_hook

    print(f"[3/3] Menganalisis Top {jumlah_clip} momen terbaik + Typography Plan menggunakan Gemini...")

    prompt = f"""
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
- end_time HARUS ditambah padding +0.20 sampai +0.85 detik setelah kata terakhir agar ending terasa lega dan tidak kepotong kasar.

ALASAN PEMILIHAN:
- Isi field 'alasan' dengan penjelasan singkat mengapa klip ini layak dipilih.
- Fokus pada nilai emosi, kekuatan hook, potensi retention, dan shareability.

ATURAN BAHASA METADATA:
- title_indonesia tetap wajib diisi untuk kompatibilitas internal / fallback.
- Semua metadata yang dipakai untuk platform harus berbahasa Inggris natural.
- Ini berlaku untuk:
  - title_inggris
  - hastag
  - description_hook
  - description_context
  - keyword_tags
  - tiktok_caption
- Jangan mencampur Bahasa Indonesia dan Bahasa Inggris di metadata platform.
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

7. tiktok_caption
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

ATURAN OUTPUT:
- Output HARUS berupa JSON array valid.
- Jangan beri penjelasan apa pun di luar JSON.
- Semua field wajib terisi.
- Jika ragu, prioritaskan akurasi isi klip daripada kreativitas berlebihan.

Transkrip:
{transkrip_lengkap}
"""

    # JSON Schema definitions
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
                    "tiktok_caption": {"type": "STRING"},
                },
                "required": [
                    "rank", "hook_start_time", "hook_end_time",
                    "start_time", "end_time", "typography_plan",
                    "broll_list", "alasan", "bgm_mood",
                    "title_indonesia", "title_inggris", "hastag",
                    "description_hook", "description_context",
                    "keyword_tags", "tiktok_caption",
                ],
            },
        },
    )

    hasil_json = _generate_json_with_retry(
        client=client,
        model=cfg.gemini_model,
        contents=prompt,
        config=gemini_config,
    )

    return hasil_json
