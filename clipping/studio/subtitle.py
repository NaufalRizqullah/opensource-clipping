"""
clipping.studio_subtitle — ASS Subtitle Builder

Berisi fungsi ``buat_file_ass()`` untuk menghasilkan file subtitle ASS
dengan dukungan advanced typography (per-word positioning, animasi bounce/stagger,
karaoke highlight) dan mode subtitle sederhana.

Dipindahkan dari studio.py tanpa perubahan logic.
"""

import os
import string

from PIL import ImageFont


def buat_file_ass(
    data_segmen,
    start_clip,
    end_clip,
    nama_file_ass,
    rasio,
    cfg,
    typography_plan=None,
    gunakan_advanced=True,
    get_x_func=None,
    source_dim=None,
):
    """Buat file subtitle ASS dari data segmen transkrip.

    Mendukung dua mode:
    - **Simple mode**: word-by-word reveal dengan karaoke highlight opsional.
    - **Advanced mode**: per-word positioning dengan typography plan (bounce_pop,
      stagger_up), font khusus, dan scaling per kata.

    Parameters
    ----------
    data_segmen : list[dict]
        Data segmen transkrip dari Whisper / YouTube subtitle.
    start_clip, end_clip : float
        Rentang waktu clip dalam detik.
    nama_file_ass : str
        Path file ASS output.
    rasio : str
        Rasio video (``"9:16"`` atau ``"16:9"``).
    cfg : SimpleNamespace
        Konfigurasi (font, alignment, margin, dll.).
    typography_plan : list[dict] | None
        Rencana tipografi per kata dari Gemini AI.
    gunakan_advanced : bool
        Gunakan advanced typography mode.
    get_x_func : callable | None
        Fungsi tracking kamera (untuk dev mode subtitle alignment).
    source_dim : tuple[int, int] | None
        Dimensi source video (width, height) untuk dev mode.
    """
    if typography_plan is None:
        typography_plan = []

    typo_dict = {}
    for plan in typography_plan:
        clean_word = plan.get("kata_utama", "").lower().strip(string.punctuation)
        typo_dict[clean_word] = plan

    pakai_advanced = cfg.use_advanced_text and gunakan_advanced
    pakai_karaoke = cfg.use_karaoke_effect

    outline_val = 3 if pakai_karaoke else 0.2
    shadow_val = 2.5 if pakai_karaoke else 0.2

    daftar_font = cfg.daftar_font
    gaya = cfg.gaya_font_aktif
    font_dir = cfg.font_dir

    font_utama_dict = daftar_font[gaya]["utama"]
    font_khusus_dict = daftar_font[gaya]["khusus"]

    font_utama = font_utama_dict["nama"]
    font_khusus = font_khusus_dict["nama"]

    scale_base_khusus = (
        cfg.scale_kata_khusus_916 if rasio == "9:16" else cfg.scale_kata_khusus_169
    )
    warna_khusus = cfg.warna_kata_khusus

    def get_scale_value(level):
        if level == 3:
            return scale_base_khusus
        elif level == 2:
            return int((scale_base_khusus + 100) / 2)
        else:
            return 110

    def fmt_time(d):
        return f"{int(d // 3600)}:{int((d % 3600) // 60):02d}:{int(d % 60):02d}.{int((d - int(d)) * 100):02d}"

    play_res_x, play_res_y = (1080, 1920) if rasio == "9:16" else (1920, 1080)
    align = cfg.ass_align_916 if rasio == "9:16" else cfg.ass_align_169
    margin_v = cfg.ass_margin_916 if rasio == "9:16" else cfg.ass_margin_169
    font_sz = cfg.ass_font_916 if rasio == "9:16" else cfg.ass_font_169
    margin_lr = 60 if rasio == "9:16" else 40

    header = (
        f"[Script Info]\n"
        f"PlayResX: {play_res_x}\n"
        f"PlayResY: {play_res_y}\n"
        f"WrapStyle: 1\n"
        f"ScriptType: v4.00+\n"
        f"ScaledBorderAndShadow: yes\n\n"
        f"[V4+ Styles]\n"
        f"Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font_utama},{font_sz},&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,{outline_val},{shadow_val},{align},{margin_lr},{margin_lr},{margin_v},1\n\n"
        f"[Events]\n"
        f"Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    def cek_font_di_folder(nama_file, min_valid_size=1000):
        path = os.path.join(font_dir, nama_file)
        return os.path.exists(path) and os.path.getsize(path) > min_valid_size

    if not pakai_advanced:
        with open(nama_file_ass, "w", encoding="utf-8") as f:
            f.write(header)
            for seg in data_segmen:
                seg_s = max(0, seg["start"] - start_clip)
                seg_e = min(end_clip - start_clip, seg["end"] - start_clip)
                if seg_s >= seg_e:
                    continue

                for i, w in enumerate(seg["words"]):
                    w_s = max(0, w["start"] - start_clip)
                    if i < len(seg["words"]) - 1:
                        w_e = min(
                            end_clip - start_clip,
                            seg["words"][i + 1]["start"] - start_clip,
                        )
                    else:
                        w_e = min(end_clip - start_clip, w["end"] - start_clip)

                    if w_s < w_e:
                        text_parts = []
                        for j, x in enumerate(seg["words"]):
                            if pakai_karaoke:
                                if j == i:
                                    text_parts.append(
                                        f"{{\\c&H00FFFF&}}{x['word']}{{\\c&HFFFFFF&}}"
                                    )
                                else:
                                    text_parts.append(x["word"])
                            else:
                                if j <= i:
                                    text_parts.append(x["word"])
                                else:
                                    text_parts.append(
                                        f"{{\\alpha&HFF&}}{x['word']}{{\\alpha&H00&}}"
                                    )

                        f.write(
                            f"Dialogue: 0,{fmt_time(w_s)},{fmt_time(w_e)},Default,,0,0,0,,{' '.join(text_parts)}\n"
                        )
        return

    # Advanced typography mode
    font_cache = {}

    def get_cached_font(is_khusus, scale_val):
        key = f"{is_khusus}_{scale_val}"
        if key not in font_cache:
            f_info = font_khusus_dict if is_khusus else font_utama_dict
            f_file = f_info["file"]
            f_path = os.path.join(font_dir, f_file)

            if not cek_font_di_folder(f_file):
                raise FileNotFoundError(f"Font tidak ditemukan: {f_path}")

            font_cache[key] = ImageFont.truetype(
                f_path,
                int(font_sz * (scale_val / 100.0)),
            )
        return font_cache[key]

    def build_font_tag(font_info):
        nama = str(font_info["nama"]).replace("{", "").replace("}", "").strip()
        bold = 1 if int(font_info.get("bold", 0)) else 0
        return f"\\fn{nama}\\b{bold}"

    max_line_width = play_res_x - (margin_lr * 2)
    space_width = font_sz * 0.25
    TIGHTNESS = 0.95

    with open(nama_file_ass, "w", encoding="utf-8") as f:
        f.write(header)

        for seg in data_segmen:
            seg_s = max(0, seg["start"] - start_clip)
            seg_e = min(end_clip - start_clip, seg["end"] - start_clip)
            if seg_s >= seg_e:
                continue

            lines = []
            current_line = []
            current_w = 0
            max_line_h = 0

            for w_dict in seg["words"]:
                word_clean = w_dict["word"].lower().strip(string.punctuation)
                plan = typo_dict.get(word_clean)

                if plan:
                    w_style = plan.get("style", "khusus")
                    w_scale = get_scale_value(plan.get("scale_level", 2))
                    is_khusus = w_style == "khusus"

                    pil_font = get_cached_font(is_khusus, w_scale)
                    raw_w = (
                        pil_font.getlength(w_dict["word"])
                        if hasattr(pil_font, "getlength")
                        else len(w_dict["word"]) * 20
                    )
                    w_len = raw_w * TIGHTNESS
                    h_len = font_sz * (w_scale / 100.0)
                else:
                    w_scale = 100
                    pil_font = get_cached_font(False, w_scale)
                    raw_w = (
                        pil_font.getlength(w_dict["word"])
                        if hasattr(pil_font, "getlength")
                        else len(w_dict["word"]) * 15
                    )
                    w_len = raw_w * TIGHTNESS
                    h_len = font_sz

                if current_line and (current_w + space_width + w_len > max_line_width):
                    lines.append(
                        {
                            "words": current_line,
                            "width": current_w,
                            "height": max_line_h,
                        }
                    )
                    current_line = []
                    current_w = 0
                    max_line_h = 0

                x_offset = current_w if not current_line else current_w + space_width
                current_line.append(
                    {
                        "text": w_dict["word"],
                        "plan": plan,
                        "w": w_len,
                        "h": h_len,
                        "x_offset": x_offset,
                        "start": max(0, w_dict["start"] - start_clip),
                        "end": min(end_clip - start_clip, w_dict["end"] - start_clip),
                    }
                )

                current_w = x_offset + w_len
                max_line_h = max(max_line_h, h_len)

            if current_line:
                lines.append(
                    {"words": current_line, "width": current_w, "height": max_line_h}
                )

            line_spacing = 15
            total_stack_h = (
                sum(l["height"] for l in lines) + (len(lines) - 1) * line_spacing
            )
            current_y = play_res_y - margin_v - total_stack_h

            for line in lines:
                start_x = (play_res_x - line["width"]) / 2
                
                if get_x_func and cfg.dev_mode and source_dim:
                    sw, sh = source_dim
                    # Calculate center of 9:16 window in source pixels
                    crop_w_src = sh * 9 // 16
                    # Reference time: midpoint of the current segment/line
                    t_ref = line["words"][0]["start"] + start_clip
                    cx = get_x_func(t_ref)
                    center_x_src = cx + (crop_w_src / 2)
                    # Target center in PlayResX (1920)
                    target_center_x = center_x_src * (play_res_x / sw)
                    start_x = target_center_x - (line["width"] / 2)

                line_y = current_y + line["height"]

                for w_data in line["words"]:
                    word_x = start_x + w_data["x_offset"] + (w_data["w"] / 2)
                    w_appear_ms = int((w_data["start"] - seg_s) * 1000)
                    w_end_ms = int((w_data["end"] - seg_s) * 1000)

                    if w_data["plan"]:
                        w_style = w_data["plan"].get("style", "khusus")
                        w_anim = w_data["plan"].get("animasi", "bounce_pop")
                        target_scale = get_scale_value(
                            w_data["plan"].get("scale_level", 2)
                        )
                        font_info = (
                            font_khusus_dict if w_style == "khusus" else font_utama_dict
                        )
                        f_tag = build_font_tag(font_info)
                        c_tag = f"\\c{warna_khusus}"
                    else:
                        w_anim = "none"
                        target_scale = 100
                        f_tag = build_font_tag(font_utama_dict)
                        c_tag = "\\c&HFFFFFF&"

                    t_start = w_appear_ms
                    t_pop = w_appear_ms + 80
                    t_settle = w_appear_ms + 150

                    if pakai_karaoke:
                        pos_tag = f"\\pos({int(word_x)},{int(line_y)})"
                        c_tag = "\\c&HFFFFFF&"
                        anim_tag = f"\\fscx{target_scale}\\fscy{target_scale}\\t({t_start},{t_start},\\c&H00FFFF&)\\t({w_end_ms},{w_end_ms},\\c&HFFFFFF&)"
                    else:
                        if w_anim == "stagger_up":
                            y_start = int(line_y + 30)
                            pos_tag = f"\\move({int(word_x)},{y_start},{int(word_x)},{int(line_y)},{t_start},{t_settle})"
                            anim_tag = f"\\alpha&HFF&\\fscx{target_scale}\\fscy{target_scale}\\t({t_start},{t_start},\\alpha&H00&)"
                        elif w_anim == "bounce_pop":
                            init_scale = int(target_scale * 0.7)
                            overshoot = int(target_scale * 1.15)
                            pos_tag = f"\\pos({int(word_x)},{int(line_y)})"
                            anim_tag = (
                                f"\\alpha&HFF&\\fscx{init_scale}\\fscy{init_scale}"
                                f"\\t({t_start},{t_start},\\alpha&H00&)"
                                f"\\t({t_start},{t_pop},\\fscx{overshoot}\\fscy{overshoot})"
                                f"\\t({t_pop},{t_settle},\\fscx{target_scale}\\fscy{target_scale})"
                            )
                        else:
                            pos_tag = f"\\pos({int(word_x)},{int(line_y)})"
                            anim_tag = f"\\alpha&HFF&\\fscx{target_scale}\\fscy{target_scale}\\t({t_start},{t_start},\\alpha&H00&)"

                    event_text = (
                        f"{{\\an2{pos_tag}{f_tag}{c_tag}{anim_tag}}}{w_data['text']}"
                    )
                    f.write(
                        f"Dialogue: 0,{fmt_time(seg_s)},{fmt_time(seg_e)},Default,,0,0,0,,{event_text}\n"
                    )

                current_y += line["height"] + line_spacing
