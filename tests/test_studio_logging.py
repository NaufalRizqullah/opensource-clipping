"""Exercise real diagnostic files without model loads, network calls or secrets."""

from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
from pathlib import Path

import pytest

from scripts import studio_logging as logs


@pytest.fixture(autouse=True)
def isolated_logs(monkeypatch, tmp_path):
    logs.close_logging()
    monkeypatch.setenv("OSC_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("OSC_LOG_RUN_ID", raising=False)
    yield
    logs.close_logging()


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_structured_event_has_correlation_and_appends_without_duplicates():
    path = logs.configure_logging()
    run_id = logs.os.environ["OSC_LOG_RUN_ID"]
    request_token = logs.current_request_id.set("request_123")
    job_token = logs.job_context.set("job_123")
    try:
        logs.log_event("stage.started", component="worker", stage="render", clip_index=2)
        assert logs.configure_logging() == path
        logs.log_event("stage.completed", component="worker", stage="render", elapsed_ms=123.45678)
    finally:
        logs.current_request_id.reset(request_token)
        logs.job_context.reset(job_token)
    logs.close_logging()
    logs.configure_logging()
    logs.log_event("server.restarted")
    events = read_events(path)
    assert len(events) == 3
    assert events[0]["job_id"] == "job_123"
    assert events[0]["request_id"] == "request_123"
    assert events[0]["clip_index"] == 2
    assert events[1]["elapsed_ms"] == 123.457
    assert events[0]["timestamp"].endswith("Z")
    assert events[0]["pid"] > 0 and events[0]["thread_id"] > 0
    assert all(event["run_id"] == run_id for event in events)


def test_roles_have_separate_files_with_same_run_id():
    supervisor = logs.configure_logging("bootstrap")
    api = logs.configure_logging("api")
    logs.log_event("setup.started", component="bootstrap", stage="setup")
    logs.log_event("api.started", component="api")
    assert supervisor.name == "kaggle-studio.jsonl"
    assert api.name == "studio-api.jsonl"
    assert read_events(supervisor)[0]["run_id"] == read_events(api)[0]["run_id"]
    assert len(read_events(supervisor)) == len(read_events(api)) == 1


def test_rotation_keeps_valid_json_and_limits_backups(monkeypatch):
    monkeypatch.setattr(logs, "MAX_BYTES", 550)
    monkeypatch.setattr(logs, "BACKUP_COUNT", 2)
    path = logs.configure_logging()
    for number in range(25):
        logs.log_event("rotation.event", count=number)
    files = list(path.parent.glob("studio-api.jsonl*"))
    assert len(files) == 3
    assert max(event["count"] for file in files for event in read_events(file)) == 24
    assert all(file.stat().st_size <= 550 for file in files)


def test_concurrent_events_do_not_interleave_or_duplicate():
    path = logs.configure_logging()
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda index: logs.log_event("thread.event", count=index), range(100)))
    events = read_events(path)
    assert len(events) == 100
    assert {event["count"] for event in events} == set(range(100))


def test_unconfigured_cli_does_not_create_log_files(monkeypatch, tmp_path):
    monkeypatch.delenv("OSC_LOG_DIR")
    monkeypatch.setattr(logs, "PROJECT_ROOT", tmp_path)
    logs.log_event("renderer.failed", error=ValueError("not copied"))
    assert not (tmp_path / "logs").exists()


def test_configuration_switch_closes_old_file_and_preserves_run_id(monkeypatch, tmp_path):
    old_path = logs.configure_logging()
    old_handler = logs._handlers["api"][2]
    logs.log_event("first.event")
    monkeypatch.setenv("OSC_LOG_DIR", str(tmp_path / "another"))
    new_path = logs.configure_logging()
    logs.log_event("second.event")
    assert old_handler.stream is None
    assert read_events(old_path)[0]["event"] == "first.event"
    assert read_events(new_path)[0]["event"] == "second.event"
    assert read_events(old_path)[0]["run_id"] == read_events(new_path)[0]["run_id"]


def test_invalid_run_id_is_replaced_without_echoing_value(monkeypatch):
    secret = "synthetic-private-value-should-never-be-logged"
    monkeypatch.setenv("OSC_LOG_RUN_ID", secret)
    path = logs.configure_logging()
    logs.log_event("safe.event")
    assert secret not in path.read_text(encoding="utf-8")
    assert len(read_events(path)[0]["run_id"]) == 32


def test_nested_error_records_locations_without_message_source_locals_or_path(capsys):
    path = logs.configure_logging()
    secret = "synthetic-private-context-not-for-logs"
    namespace = {"__name__": "probe.safe_module"}
    code = compile(
        "def fail():\n"
        f"    local_secret = {secret!r}\n"
        "    try:\n"
        "        raise ImportError(local_secret)\n"
        "    except ImportError as exc:\n"
        "        raise ValueError(local_secret) from exc\n",
        f"/private/{secret}/secret_module.py", "exec",
    )
    exec(code, namespace)
    try:
        namespace["fail"]()
    except ValueError as error:
        logs.log_event("job.failed", component="worker", level="ERROR", stage="render", error=error)
    content = path.read_text(encoding="utf-8")
    event = json.loads(content)
    assert secret not in content
    assert "local_secret" not in content and "/private/" not in content
    assert event["error"]["type"] == "ValueError"
    assert event["error"]["traceback"][-1] == {"module": "probe.safe_module", "function": "fail", "line": 6}
    assert event["error"]["causes"][0]["type"] == "ImportError"
    assert event["error"]["causes"][0]["code"] == "import_error"
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("text,code", [
    ("numpy.dtype size changed private text", "numpy_abi_mismatch"),
    ("module://matplotlib_inline.backend_inline is not a valid value for backend", "matplotlib_backend"),
])
def test_error_classification_uses_fixed_codes_only(text, code):
    path = logs.configure_logging()
    logs.log_event("readiness.failed", error=ValueError(text))
    assert read_events(path)[0]["error"]["code"] == code
    assert text not in path.read_text(encoding="utf-8")


def test_readiness_report_drops_arbitrary_messages_and_fields():
    path = logs.configure_logging("bootstrap")
    private = "synthetic private provider response"
    logs.log_event("readiness.failed", component="bootstrap", diagnostics={
        "ready": False, "raw_output": private,
        "checks": {"renderer": {
            "ok": False, "module": "mediapipe.tasks.python.vision", "error": "ValueError",
            "code": "import_failed", "message": private, "locals": {"token": private},
            "traceback": [{"module": "matplotlib", "function": "<module>", "line": 1292,
                           "filename": private, "source": private}],
            "causes": [{"error": "ImportError", "message": private}],
        }, private: {"ok": False}},
        "versions": {"numpy": "1.26.4", "torch": private, private: "1.0"},
    })
    content = path.read_text(encoding="utf-8")
    report = read_events(path)[0]["diagnostics"]
    assert private not in content
    assert report["checks"]["renderer"]["traceback"][0]["line"] == 1292
    assert report["versions"] == {"numpy": "1.26.4"}


def test_exception_cause_cycles_are_bounded():
    first, second = ValueError("first"), RuntimeError("second")
    first.__cause__, second.__cause__ = second, first
    path = logs.configure_logging()
    logs.log_event("cyclic.error", error=first)
    assert len(read_events(path)[0]["error"]["causes"]) == 1


def test_subprocess_and_os_errors_preserve_numeric_codes_without_arguments():
    path = logs.configure_logging()
    private = "synthetic-private-command-and-path"
    logs.log_event("renderer_failed", component="renderer", clip_rank=7,
                   error=subprocess.CalledProcessError(2, ["ffmpeg", private], stderr=private))
    logs.log_event("file_failed", error=OSError(28, private, private))
    events = read_events(path)
    assert events[0]["error"]["exit_code"] == 2
    assert events[0]["clip_rank"] == 7
    assert events[1]["error"]["errno"] == 28
    assert private not in path.read_text(encoding="utf-8")


def test_open_failure_does_not_mask_application_error(monkeypatch, tmp_path, capsys):
    blocked = tmp_path / "regular-file"
    blocked.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("OSC_LOG_DIR", str(blocked))
    path = logs.configure_logging()
    logs.log_event("job.failed", error=ValueError("private context"))
    logs.log_event("job.failed", error=RuntimeError("another private context"))
    assert path == blocked / "studio-api.jsonl"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("Studio diagnostic file is unavailable") == 1
    assert "private context" not in captured.err


def test_disk_write_failure_is_nonfatal_and_never_prints_record(monkeypatch, capsys):
    logs.configure_logging()
    handler = logs._handlers["api"][2]

    class FullDisk:
        def write(self, value):
            raise OSError("synthetic private disk detail")
        def flush(self):
            pass
        def close(self):
            pass

    handler.stream.close()
    monkeypatch.setattr(handler, "stream", FullDisk())
    monkeypatch.setattr(handler, "shouldRollover", lambda record: False)
    logs.log_event("job.failed", error=ValueError("synthetic private exception"))
    logs.log_event("job.failed")
    output = capsys.readouterr().err
    assert output.count("Studio diagnostic file is unavailable") == 1
    assert "synthetic" not in output and "job.failed" not in output


def test_malformed_optional_fields_are_ignored_without_invalid_json():
    path = logs.configure_logging()
    logs.log_event("safe.event", elapsed_ms=float("nan"), count=True, route="/api?token=private",
                   diagnostics={"checks": {"renderer": {"error": [], "code": {}, "module": []}}})
    event = read_events(path)[0]
    assert "elapsed_ms" not in event and "count" not in event and "route" not in event
    assert event["diagnostics"] == {"checks": {"renderer": {}}}


def test_unmatched_route_keeps_fixed_sentinel_without_request_path():
    path = logs.configure_logging()
    logs.log_event("request.completed", method="GET", route="unmatched", status_code=404)
    assert read_events(path)[0]["route"] == "unmatched"


class PyngrokNgrokHTTPError(Exception):
    """Mirror pyngrok's public exception fields without a runtime dependency."""

    __module__ = "pyngrok.exception"

    def __init__(self, error, url, status_code, message, headers, body):
        super().__init__(error)
        self.url = url
        self.status_code = status_code
        self.message = message
        self.headers = headers
        self.body = body


def test_ngrok_http_failure_records_codes_without_response_or_credentials():
    private = "synthetic-private-response-token"
    error = PyngrokNgrokHTTPError(
        "tunnel creation failed " + private,
        "http://private-host/api/endpoints?token=" + private,
        400, private, {"Authorization": private},
        json.dumps({"error_code": 103, "msg": private,
                    "details": {"err": "ERR_NGROK_108 " + private}}),
    )
    path = logs.configure_logging("bootstrap")
    logs.log_event("tunnel.failed", component="bootstrap", error=error,
                   diagnostics={"versions": {"pyngrok": "9.1.3"}})
    event = read_events(path)[0]
    assert event["error"]["http_status"] == 400
    assert event["error"]["ngrok_error_code"] == "ERR_NGROK_108"
    assert event["error"]["agent_error_code"] == 103
    assert event["diagnostics"]["versions"] == {"pyngrok": "9.1.3"}
    text = path.read_text(encoding="utf-8")
    assert private not in text and "private-host" not in text
    assert "Authorization" not in text and "details" not in text


@pytest.mark.parametrize("source", ["ngrok_error_code", "ngrok_error", "message"])
def test_ngrok_codes_support_new_and_legacy_exception_fields(source):
    error = PyngrokNgrokHTTPError("failed", "private URL", 502, None, {}, "")
    if source == "message":
        error.args = ("authentication failed ERR_NGROK_4018 private value",)
    elif source == "ngrok_error_code":
        error.ngrok_error_code = "ERR_NGROK_4018"
    else:
        error.ngrok_error = "authentication failed ERR_NGROK_4018 private value"
    assert logs.external_error_metadata(error) == {
        "http_status": 502, "ngrok_error_code": "ERR_NGROK_4018",
    }


@pytest.mark.parametrize("status,body,code", [
    (True, [], []),
    ("400", {"error_code": 103}, {}),
    (600, '{"error_code": true}', "ERR_NGROK_123_PRIVATE"),
    (99, '{"error_code": "103"}', "ERR_NGROK_12345678901"),
    (None, '[' * 1000 + ']' * 1000, "private value"),
])
def test_ngrok_malformed_metadata_is_omitted(status, body, code):
    error = PyngrokNgrokHTTPError("failure", "private URL", status, None, {}, body)
    error.ngrok_error_code = code
    assert logs.external_error_metadata(error) == {}


def test_ngrok_oversized_response_is_not_parsed_or_searched():
    body = '{"error_code":103,"msg":"' + "x" * logs._ERROR_TEXT_LIMIT + ' ERR_NGROK_108"}'
    error = PyngrokNgrokHTTPError("failure", "private URL", 400, None, {}, body)
    assert logs.external_error_metadata(error) == {"http_status": 400}


def test_provider_metadata_ignores_ngrok_fields_on_unrelated_errors():
    error = RuntimeError("ERR_NGROK_108 private value")
    error.status_code = 429
    error.ngrok_error_code = "ERR_NGROK_108"
    error.body = '{"error_code":103}'
    assert logs.external_error_metadata(error) == {"http_status": 429}


def test_provider_property_failure_does_not_mask_original_exception():
    class PropertyFailure(Exception):
        @property
        def status_code(self):
            raise ValueError("private property value")

    assert logs.external_error_metadata(PropertyFailure()) == {}
