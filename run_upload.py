#!/usr/bin/env python3
"""
run_upload.py — CLI Entry Point for YouTube Auto-Uploader & Scheduler

Usage:
    python run_upload.py                       # Defaults (safe mode)
    python run_upload.py --test-mode           # Only upload 1 video
    python run_upload.py --interval-hours 48
    python run_upload.py --no-approval         # Skip manual approval (risky)
"""

import sys
import os
import argparse

from youtube_uploader import upload_manifest_to_youtube
from youtube_uploader.safety import load_safety_config


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="🚀 OpenSource Clipping — YouTube Auto-Uploader",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--token-file", default=".credentials/youtube_token.json",
                   help="Path to YouTube OAuth token (JSON format)")
    p.add_argument("--manifest-file", default="outputs/render_manifest.json",
                   help="Input manifest file from clipping pipeline")
    p.add_argument("--result-file", default="outputs/youtube_upload_results.json",
                   help="Output JSON trace file for upload responses")
    p.add_argument("--updated-manifest", default="outputs/render_manifest_uploaded.json",
                   help="Output upgraded manifest file")
    p.add_argument("--tz-name", default="Asia/Makassar",
                   help="Timezone for scheduling (IANA format)")
    p.add_argument("--interval-hours", type=int, default=24,
                   help="Delay interval between uploads (minimum enforced by safety config)")
    p.add_argument("--start-local", default=None,
                   help="Manual start time bypass (format: YYYY-MM-DD HH:MM)")
    p.add_argument("--test-mode", action="store_true",
                   help="Only upload the FIRST item in the manifest for testing purposes")

    # Safety options
    safety_group = p.add_argument_group("Safety Options")
    safety_group.add_argument(
        "--safety-config", default="upload_safety.json",
        help="Path to upload safety config JSON file",
    )
    safety_group.add_argument(
        "--no-approval", action="store_true",
        help="Skip manual approval prompts (⚠️ risky — not recommended for channels recovering from ban)",
    )

    return p


def main():
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:])

    print("=" * 70)
    print("🚀 YouTube Uploader")
    print("=" * 70)

    # Ensure credentials dir exists
    creds_dir = os.path.dirname(args.token_file)
    if creds_dir:
        os.makedirs(creds_dir, exist_ok=True)

    if not os.path.exists(args.token_file):
        print(f"❌ ERROR: File kredensial tidak ditemukan di '{args.token_file}'.")
        print(f"   Mohon tempatkan file 'youtube_token.json' di dalam folder '{creds_dir}'.")
        sys.exit(1)

    # Load safety config
    safety_config = load_safety_config(args.safety_config)

    if args.no_approval:
        print("\n⚠️  WARNING: Manual approval dinonaktifkan (--no-approval).")
        print("   Semua video akan langsung diupload tanpa konfirmasi.")
        print("   Ini TIDAK DIREKOMENDASIKAN untuk channel yang pernah kena ban.\n")

    upload_manifest_to_youtube(
        token_file=args.token_file,
        manifest_file=args.manifest_file,
        result_file=args.result_file,
        updated_manifest_file=args.updated_manifest,
        tz_name=args.tz_name,
        interval_hours=args.interval_hours,
        start_local=args.start_local,
        test_mode=args.test_mode,
        safety_config=safety_config,
        skip_approval=args.no_approval,
    )

    print("\n✅ Proses upload YouTube selesai.")


if __name__ == "__main__":
    main()
