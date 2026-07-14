#!/usr/bin/env python3
"""
run_fb_upload.py — CLI Entry Point for Facebook Pages Reels Auto-Uploader

Usage:
    python run_fb_upload.py                       # Defaults
    python run_fb_upload.py --test-mode           # Only upload 1 video
    python run_fb_upload.py --interval-hours 3    # 3 hour gap between videos
"""

import sys
import os
import argparse

from facebook_uploader import upload_manifest_to_facebook


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OpenSource Clipping -- Facebook Pages Reels Auto-Uploader",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Determine timezone default from env
    default_tz = os.environ.get("APP_TIMEZONE", "Asia/Makassar").strip()

    p.add_argument("--manifest-file", default="outputs/render_manifest.json",
                   help="Input manifest file from clipping pipeline")
    p.add_argument("--result-file", default="outputs/fb_upload_results.json",
                   help="Output JSON trace file for upload responses")
    p.add_argument("--updated-manifest", default="outputs/render_manifest_fb_uploaded.json",
                   help="Output upgraded manifest file")
    p.add_argument("--tz-name", default=default_tz,
                   help="Timezone for scheduling (IANA format)")
    p.add_argument("--interval-hours", type=int, default=5,
                   help="Delay interval between scheduled uploads (in hours)")
    p.add_argument("--test-mode", action="store_true",
                   help="Only upload the FIRST item in the manifest for testing purposes")

    return p


def main():
    # Load .env early so META_* vars are available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:])

    print("=" * 70)
    print("🚀 Facebook Pages Reels Uploader")
    print("=" * 70)

    # Check required env vars
    if not os.environ.get("META_PAGE_ACCESS_TOKEN"):
        print("❌ ERROR: META_PAGE_ACCESS_TOKEN belum di-set.")
        print("   Tambahkan ke file .env atau set sebagai environment variable.")
        sys.exit(1)

    if not os.environ.get("META_PAGE_ID"):
        print("❌ ERROR: META_PAGE_ID belum di-set.")
        print("   Tambahkan ke file .env atau set sebagai environment variable.")
        sys.exit(1)

    upload_manifest_to_facebook(
        manifest_file=args.manifest_file,
        result_file=args.result_file,
        updated_manifest_file=args.updated_manifest,
        tz_name=args.tz_name,
        interval_hours=args.interval_hours,
        test_mode=args.test_mode,
    )

    print("\n✅ Proses upload Facebook selesai.")


if __name__ == "__main__":
    main()
