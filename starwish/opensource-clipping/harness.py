"""Deterministic Starwish harness. Video and output files stay on the workspace VM."""

import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

from clip_ranges import normalize_ranges


WORKSPACE_COMMAND = "python3 /usr/local/lib/opensource-clipping/workspace_job.py"


def requested_video(params):
    if not isinstance(params, dict) or params.keys() - {"youtube_url", "ranges"}:
        raise ValueError("Supported parameters are youtube_url and ranges.")
    if "youtube_url" not in params:
        return None
    url = params["youtube_url"]
    if not isinstance(url, str):
        raise ValueError("youtube_url must be a public YouTube video URL.")
    parsed = urlsplit(url)
    host = parsed.netloc.lower()
    if parsed.scheme == "https" and host == "youtu.be":
        video_id = parsed.path.removeprefix("/")
    elif parsed.scheme == "https" and host in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            ids = parse_qs(parsed.query).get("v", [])
            video_id = ids[0] if len(ids) == 1 else ""
        else:
            video_id = parsed.path.removeprefix("/shorts/") if parsed.path.startswith("/shorts/") else ""
    else:
        video_id = ""
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Use an HTTPS YouTube watch, shorts or youtu.be link to one video.")
    return video_id


def rpc_reply(response, call_id):
    if response.headers.get_content_type() == "text/event-stream":
        data = []
        for raw in response:
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
            elif not line and data:
                message = json.loads("\n".join(data))
                if message.get("id") == call_id:
                    return message
                data.clear()
        raise RuntimeError("The workspace stream ended before the tool answered.")
    return json.load(response)


class Workspace:
    def __init__(self):
        self.url = os.environ["STARWISH_MCP_URL"]
        self.headers = {
            "Content-Type": "application/json", "Accept": "application/json, text/event-stream",
            "Authorization": "Bearer " + os.environ["STARWISH_MCP_TOKEN"],
        }
        self.sequence = 0

    def rpc(self, method, params, *, notification=False):
        packet = {"jsonrpc": "2.0", "method": method, "params": params}
        if not notification:
            self.sequence += 1
            packet["id"] = self.sequence
        request = Request(self.url, json.dumps(packet).encode(), self.headers, method="POST")
        with urlopen(request, timeout=930) as response:
            if session := response.headers.get("Mcp-Session-Id"):
                self.headers["Mcp-Session-Id"] = session
            if notification:
                return
            reply = rpc_reply(response, self.sequence)
        if reply.get("id") != self.sequence or "result" not in reply or "error" in reply:
            raise RuntimeError(f"Workspace RPC failed during {method}.")
        return reply["result"]

    def render(self, video_id, ranges=None):
        init = self.rpc("initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "opensource-clipping", "version": "0.2.0"},
        })
        self.headers["MCP-Protocol-Version"] = init["protocolVersion"]
        self.rpc("notifications/initialized", {}, notification=True)
        available = self.rpc("tools/list", {})
        if not any(tool["name"] == "run_command" for tool in available["tools"]):
            raise RuntimeError("The workspace does not offer run_command.")
        payload = {"video_id": video_id}
        if ranges is not None:
            payload["ranges"] = ranges
        result = self.rpc("tools/call", {"name": "run_command", "arguments": {
            "command": WORKSPACE_COMMAND, "timeout_seconds": 900,
            "stdin": json.dumps(payload),
        }})
        text = "\n".join(part["text"] for part in result.get("content", []) if part.get("type") == "text")
        if result.get("isError") or "STARWISH_CPU_CLIP_COMPLETE" not in text:
            raise RuntimeError("Workspace render did not finish. Check the workspace command log and input media.")


def main():
    request_file = Path(os.environ.get("STARWISH_REQUEST_FILE", "/job/request.json"))
    job = json.loads(request_file.read_text(encoding="utf-8"))
    params = job.get("params")
    params = {} if params is None else params
    video_id = requested_video(params)
    ranges = normalize_ranges(params["ranges"]) if "ranges" in params else None
    Workspace().render(video_id, ranges)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        sys.exit(f"Job failed: {exc}")
