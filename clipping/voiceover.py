"""
clipping.voiceover — AI Commentary & TTS Generation Module

Handles generation of commentary scripts using Gemini AI and converts them
to speech using edge-tts. Supports word-level subtitle generation.
"""

import os
import json
import asyncio
from google import genai
from google.genai import types

try:
    import edge_tts
except ImportError:
    edge_tts = None


# ==============================================================================
# TTS SYNTHESIS (EDGE-TTS)
# ==============================================================================

async def _synthesize_async(text: str, voice: str, output_audio_path: str, output_subs_path: str = None):
    """Async core for TTS generation."""
    if edge_tts is None:
        raise ImportError("edge-tts is not installed. Run: pip install edge-tts")

    communicate = edge_tts.Communicate(text, voice)
    submaker = edge_tts.SubMaker()

    with open(output_audio_path, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                submaker.feed(chunk)

    if output_subs_path:
        # Save SRT for reference (edge-tts v7+ uses get_srt instead of generate_subs)
        with open(output_subs_path, "w", encoding="utf-8") as file:
            file.write(submaker.get_srt())
            
        # Extract timings directly from submaker.cues instead of parsing text files
        segments = []
        for cue in submaker.cues:
            start_s = cue.start.total_seconds()
            end_s = cue.end.total_seconds()
            word = cue.content
            segments.append({
                "start": start_s,
                "end": end_s,
                "text": word,
                "words": [{
                    "word": word,
                    "start": start_s,
                    "end": end_s,
                    "probability": 1.0
                }]
            })
        return segments
    return []

def synthesize_voice(text: str, voice: str, output_dir: str, clip_id: str) -> tuple[str, list[dict]]:
    """
    Synthesize text to speech using edge-tts.
    
    Returns:
        tuple[str, list[dict]]: (path_to_audio_mp3, list_of_subtitle_segments)
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, f"vo_clip_{clip_id}.mp3")
    subs_path = os.path.join(output_dir, f"vo_clip_{clip_id}.vtt")
    
    print(f"   🎙️ Synthesizing voice-over ({voice}) for clip {clip_id}...")
    segments = asyncio.run(_synthesize_async(text, voice, audio_path, subs_path))
    
    # We consolidate the word-level segments into 3-word chunks for better subtitle rendering
    consolidated = _consolidate_segments(segments, words_per_seg=3)
    
    return audio_path, consolidated

def _consolidate_segments(raw_segments: list[dict], words_per_seg: int = 3) -> list[dict]:
    """Group word-level VTT segments into chunked segments for better readability."""
    if not raw_segments:
        return []
        
    consolidated = []
    current_chunk = {"start": 0.0, "end": 0.0, "text": "", "words": []}
    word_count = 0
    
    for i, seg in enumerate(raw_segments):
        if not seg["words"]:
            continue
            
        word_data = seg["words"][0]
        
        if word_count == 0:
            current_chunk["start"] = word_data["start"]
            
        current_chunk["words"].append(word_data)
        current_chunk["end"] = word_data["end"]
        word_count += 1
        
        if word_count >= words_per_seg or i == len(raw_segments) - 1:
            current_chunk["text"] = " ".join(w["word"] for w in current_chunk["words"])
            consolidated.append(current_chunk)
            current_chunk = {"start": 0.0, "end": 0.0, "text": "", "words": []}
            word_count = 0
            
    return consolidated


# ==============================================================================
# AI COMMENTARY SCRIPT GENERATION
# ==============================================================================

def get_commentary_prompt(transcript_snippet: str, style: str, language: str, length: str) -> str:
    lang_instruction = "Gunakan bahasa Indonesia yang gaul tapi profesional (seperti narator YouTube/TikTok)."
    if language == "en":
        lang_instruction = "Use engaging, conversational English suitable for a YouTube/TikTok narrator."
        
    style_instructions = {
        "analysis": "Berikan analisis tajam atau opini insightfull tentang kenapa momen ini penting atau menarik.",
        "reaction": "Berikan reaksi natural seolah kamu sedang menonton momen ini dan terkesan/terkejut.",
        "lesson": "Tarik satu pelajaran atau 'moral of the story' yang bisa diaplikasikan penonton dari momen ini.",
        "summary": "Berikan konteks atau ringkasan singkat tapi memikat tentang apa yang terjadi di momen ini."
    }
    
    if language == "en":
        style_instructions = {
            "analysis": "Provide a sharp analysis or insightful opinion on why this moment is important or interesting.",
            "reaction": "Provide a natural reaction as if you are watching this moment and are impressed/surprised.",
            "lesson": "Extract one key lesson or takeaway that the audience can apply from this moment.",
            "summary": "Provide a catchy but brief context or summary of what's happening in this moment."
        }
        
    chosen_style = style_instructions.get(style, style_instructions["analysis"])

    length_instruction = "berdurasi pendek (3-5 kalimat, sekitar 20-40 detik saat diucapkan)"
    if length == "short":
        length_instruction = "berdurasi sangat pendek (1-2 kalimat, sekitar 5-15 detik saat diucapkan)"
    elif length == "long":
        length_instruction = "berdurasi lumayan panjang (5-7 kalimat, sekitar 40-60 detik saat diucapkan)"
    if language == "en":
        length_instruction = "short duration (3-5 sentences, around 20-40 seconds when spoken)"
        if length == "short":
            length_instruction = "very short duration (1-2 sentences, around 5-15 seconds when spoken)"
        elif length == "long":
            length_instruction = "medium duration (5-7 sentences, around 40-60 seconds when spoken)"

    prompt = f"""Kamu adalah seorang narator/komentator video pendek (Shorts/TikTok/Reels).
Tugasmu adalah membuat script voice-over {length_instruction} 
berdasarkan transkrip video berikut.

{lang_instruction}
{chosen_style}

ATURAN:
1. JANGAN sekadar mengulang isi transkrip. Tambahkan value/opini/konteks kamu sendiri.
2. JANGAN menggunakan sapaan pembuka seperti "Halo guys" atau penutup seperti "Jangan lupa subscribe". Langsung to the point ke isi momen.
3. JANGAN berikan elemen format atau penjelasan tambahan. HANYA KELUARKAN TEKS SCRIPT YANG AKAN DIBACAKAN.

TRANSKRIP KLIP:
\"\"\"
{transcript_snippet}
\"\"\"

SCRIPT VOICE-OVER (Hanya teks yang dibacakan, tanpa tanda kutip di awal/akhir):"""
    return prompt

def generate_commentary_script(transcript_snippet: str, cfg, style="analysis", language="id", length="short") -> str:
    """Generate commentary script using Gemini AI."""
    print(f"   🧠 Generating {style} commentary script via Gemini ({language}, {length})...")
    
    api_key = cfg.api_key_gemini
    if not api_key:
        raise ValueError("GOOGLE_API_KEY tidak ditemukan di environment atau config.")

    client = genai.Client(api_key=api_key)
    prompt = get_commentary_prompt(transcript_snippet, style, language, length)
    
    gemini_config = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.9,
    )
    
    model = getattr(cfg, "gemini_model", "gemini-3-flash-preview")
    fallback = getattr(cfg, "gemini_fallback_model", "gemini-2.5-flash")
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=gemini_config
        )
        if response.text:
            script = response.text.strip().strip('"').strip()
            print(f"   ✅ Script generated ({len(script)} chars)")
            return script
    except Exception as e:
        print(f"   ⚠️ Gemini main model failed: {e}. Trying fallback...")
        try:
            response = client.models.generate_content(
                model=fallback,
                contents=prompt,
                config=gemini_config
            )
            if response.text:
                script = response.text.strip().strip('"').strip()
                print(f"   ✅ Script generated via fallback ({len(script)} chars)")
                return script
        except Exception as e2:
            print(f"   ❌ Gemini fallback failed: {e2}")
            
    return ""
