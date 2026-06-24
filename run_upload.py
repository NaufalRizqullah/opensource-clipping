#!/usr/bin/env python3
"""
run_upload.py - CLI Entry Point for YouTube Auto-Uploader & Scheduler

Usage:
    python run_upload.py                       # Defaults
    python run_upload.py --test-mode           # Only upload 1 video
    python run_upload.py --interval-hours 12
"""

import sys
import os
import argparse

from youtube_uploader import upload_manifest_to_youtube


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="🚀 OpenSource Clipping - YouTube Auto-Uploader",
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
    p.add_argument("--interval-hours", type=int, default=8,
                   help="Delay interval between uploads")
    p.add_argument("--start-local", default=None,
                   help="Manual start time bypass (format: YYYY-MM-DD HH:MM)")
    p.add_argument("--test-mode", action="store_true",
                   help="Only upload the FIRST item in the manifest for testing purposes")

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
        print(f"❌ ERROR: Credentials file not found at '{args.token_file}'.")
        print(f"   Please place the 'youtube_token.json' file inside the '{creds_dir}' folder.")
        sys.exit(1)

    upload_manifest_to_youtube(
        token_file=args.token_file,
        manifest_file=args.manifest_file,
        result_file=args.result_file,
        updated_manifest_file=args.updated_manifest,
        tz_name=args.tz_name,
        interval_hours=args.interval_hours,
        start_local=args.start_local,
        test_mode=args.test_mode
    )

    print("\n✅ YouTube upload process complete.")


if __name__ == "__main__":
    main()
