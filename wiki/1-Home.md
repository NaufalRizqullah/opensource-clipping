# 🎬 OpenSource Clipping — Wiki

Welcome to the **OpenSource Clipping** wiki! This is the central hub for all documentation related to the project.

**OpenSource Clipping** is an open-source AI-powered content factory that transforms long-form videos into cinematic short-form highlights with hook teasers, karaoke subtitles, and auto-thumbnails.

---

## 📚 Table of Contents

| # | Page | Description |
|---|---|---|
| 1 | **[Home](Home)** | You are here — project overview and navigation |
| 2 | **[Getting Started](Getting-Started)** | Installation, prerequisites, API keys setup |
| 3 | **[CLI Reference](CLI-Reference)** | Full list of all command-line arguments |
| 4 | **[Face Tracking & Auto-Framing](Face-Tracking-and-Auto-Framing)** | How AI face detection and camera tracking works |
| 5 | **[Podcast Modes](Podcast-Modes)** | Split-screen & camera-switch for multi-speaker content |
| 6 | **[Hook V2 & Segment Trimming](Hook-V2-and-Segment-Trimming)** | Multi-hook intros and AI-driven clip trimming |
| 7 | **[Subtitles & Typography](Subtitles-and-Typography)** | Karaoke subtitles, font styles, kinetic text |
| 8 | **[BGM & Audio](BGM-and-Audio)** | Background music, ducking, and audio settings |
| 9 | **[Video Quality & Rendering](Video-Quality-and-Rendering)** | Resolution, bitrate, sharpening, encoder tuning |
| 10 | **[Story Clip Mode](Story-Clip-Mode)** | Multi-source narrative assembly for campaigns |
| 11 | **[YouTube Auto-Upload](YouTube-Auto-Upload)** | Automated uploading with scheduling support |
| 12 | **[Google Colab Guide](Google-Colab-Guide)** | Running the pipeline on Google Colab (free GPU) |
| 13 | **[Troubleshooting & FAQ](Troubleshooting-and-FAQ)** | Common errors, fixes, and frequently asked questions |
| 14 | **[Contributing](Contributing)** | How to contribute, report issues, and support the project |
| 15 | **[YouTube API Setup Guide](YouTube-API-Setup-Guide)** | Step-by-step guide to create OAuth credentials for auto-upload |
| 16 | **[Facebook Reels Uploader](Facebook-Reels-Uploader)** | Automated Reels uploader and scheduler for Facebook Pages |
| 17 | **[YouTube Tracker](YouTube-Tracker)** | Local playlist snapshot tracker for clipping source material |

---

## ✨ Key Features at a Glance

- 🤖 **AI Transcriber** — Word-level transcription using Faster-Whisper (large-v3)
- 🧠 **AI Content Curator** — Google Gemini analyzes context and picks the most viral moments
- 🎯 **Smart Auto-Framing** — Face-tracking via MediaPipe / YOLOv8 with smooth pan and anti-jitter
- 🎬 **Cinematic Teaser Hook** — 3-second hook with dark overlay and TV Glitch transition
- 📝 **Karaoke Subtitles** — Word-by-word highlighted `.ASS` subtitles (Hormozi / Veed style)
- 🎥 **B-Roll Integration** — Auto-fetches contextual stock footage from Pexels
- 🎙️ **Podcast Split-Screen** — Auto speaker diarization with top-bottom split layout
- 📤 **Auto YouTube Upload** — Upload with scheduling and full metadata support
- 📐 **5 Aspect Ratios** — `9:16`, `16:9`, `1:1`, `3:4`, `4:5`

## 🔄 Pipeline Flow

```
Video URL → Download → Whisper Transcription → Gemini AI Analysis → Metadata QA → Render Loop
                                                                                      ↓
                                                            Face-Track Crop + B-Roll + BGM + Subtitles + Hook
                                                                                      ↓
                                                                              Final MP4 + Thumbnail
```

## 📊 Results

| 1 video 9:16 | 1 video split from YouTube Shorts link |
|:---:|:---:|
| <a href="https://www.youtube.com/shorts/hTtU4iI-aKA"><img src="https://img.youtube.com/vi/hTtU4iI-aKA/0.jpg" width="250"></a><br>*(Standard 9:16 Auto-Framing)* | <a href="https://www.youtube.com/shorts/RnoJqC8Yur4"><img src="https://img.youtube.com/vi/RnoJqC8Yur4/0.jpg" width="250"></a><br>*(Split-Screen Podcast Mode)* |

---

## 🔗 Quick Links

- [GitHub Repository](https://github.com/NaufalRizqullah/opensource-clipping)
- [Changelog](https://github.com/NaufalRizqullah/opensource-clipping/blob/main/CHANGELOG.md)
- [Story Clip Documentation](https://github.com/NaufalRizqullah/opensource-clipping/blob/main/docs/STORY_CLIP.md)
