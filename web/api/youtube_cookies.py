"""YouTube cookies for the current Studio backend. Only status is public."""

import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit


MAX_BYTES = 1024 * 1024
_uploaded: bytes | None = None
_YOUTUBE_HOST = re.compile(r"(?:[a-z0-9-]+\.)*(?:youtube\.com|youtu\.be)", re.ASCII)


def operator_file(project: Path) -> Path | None:
    configured = os.environ.get("OSC_YOUTUBE_COOKIES_FILE")
    path = Path(configured) if configured else project / "youtube_cookies.txt"
    return path if path.is_file() else None


def status(project: Path) -> dict[str, bool]:
    uploaded = _uploaded is not None
    return {
        "youtube_cookies_set": uploaded or operator_file(project) is not None,
        "youtube_cookies_uploaded": uploaded,
    }


def _normalize(data: bytes) -> bytes:
    if len(data) > MAX_BYTES:
        raise ValueError("The YouTube cookies file exceeds 1 MiB.")
    invalid = "Choose a Netscape cookies.txt export with seven tab-separated fields per cookie."
    try:
        lines = data.decode("utf-8-sig").replace("\r\n", "\n").split("\n")
    except UnicodeDecodeError:
        raise ValueError(invalid) from None
    if lines[0] not in {"# Netscape HTTP Cookie File", "# HTTP Cookie File"}:
        raise ValueError(invalid)

    selected = ["# Netscape HTTP Cookie File"]
    now = time.time()
    for line in lines[1:]:
        if not line.strip() or (line.startswith("#") and not line.startswith("#HttpOnly_")):
            continue
        fields = line.removeprefix("#HttpOnly_").split("\t")
        if len(fields) != 7:
            raise ValueError(invalid)
        domain, subdomains, path, secure, expires, name, value = fields
        if not _YOUTUBE_HOST.fullmatch(domain.removeprefix(".").lower()):
            continue
        if (subdomains not in {"TRUE", "FALSE"} or (subdomains == "TRUE") != domain.startswith(".")
                or secure not in {"TRUE", "FALSE"} or not path.startswith("/")
                or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name)
                or not re.fullmatch(r"[0-9]{0,15}", expires)
                or any(ord(c) < 32 or ord(c) == 127 for c in path + value)):
            raise ValueError(invalid)
        timestamp = int(expires or 0)
        if not timestamp or timestamp > now:
            selected.append(line)
    if len(selected) == 1:
        raise ValueError("No current YouTube cookies found. Export cookies for youtube.com again.")
    return ("\n".join(selected) + "\n").encode("utf-8")


def replace(data: bytes) -> None:
    global _uploaded
    _uploaded = _normalize(data)  # Keep the previous upload if validation fails.


def clear() -> None:
    global _uploaded
    _uploaded = None


def for_download(platform: str, url: str, fallback_file: str | None) -> bytes | None:
    if platform != "youtube":
        return None
    try:
        target = urlsplit(url)
        host = target.hostname or ""
    except ValueError:
        return None
    if target.scheme not in {"http", "https"} or not _YOUTUBE_HOST.fullmatch(host):
        return None
    uploaded = _uploaded  # Immutable snapshot for the whole download, including subtitles.
    if uploaded is not None:
        return uploaded
    if fallback_file:
        with Path(fallback_file).open("rb") as source:
            return _normalize(source.read(MAX_BYTES + 1))
    return None
