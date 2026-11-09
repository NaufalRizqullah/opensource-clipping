"""Managed ASGI entry point that records failures before FastAPI can start."""

from __future__ import annotations

import argparse
import time

from .studio_logging import configure_logging, log_event


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    configure_logging("api")
    stage = "api_import"
    started = time.monotonic()
    log_event("server.import.started", component="api", stage=stage)
    try:
        import uvicorn
        from web.api.app import app

        log_event("server.import.completed", component="api", stage=stage,
                  elapsed_ms=(time.monotonic() - started) * 1000)
        stage = "api_serve"
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except KeyboardInterrupt:
        log_event("server.cancelled", component="api", stage=stage)
        return 130
    except (Exception, SystemExit) as exc:
        code = exc.code if isinstance(exc, SystemExit) and type(exc.code) is int else 1
        log_event("server.failed" if code else "server.stopped", component="api", stage=stage,
                  level="ERROR" if code else "INFO", error=exc, exit_code=code)
        return code
    log_event("server.stopped", component="api", stage=stage, exit_code=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
