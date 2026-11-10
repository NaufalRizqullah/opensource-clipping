"""Exercise the real pyngrok SDK offline, stopping before agent/network access."""

import json
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from scripts import kaggle_studio as bootstrap
from scripts import studio_logging


@pytest.fixture
def sdk(monkeypatch, tmp_path):
    # Set synthetic credentials before pyngrok creates its default config.
    monkeypatch.setenv("NGROK_AUTHTOKEN", "synthetic-private-token")
    monkeypatch.delenv("NGROK_API_KEY", raising=False)
    from pyngrok import conf, ngrok

    studio_logging.close_logging()
    monkeypatch.setenv("OSC_LOG_DIR", str(tmp_path / "logs"))
    logfile = studio_logging.configure_logging("bootstrap")
    state = SimpleNamespace(requests=[], killed=[], configs=[], public_url="https://example.test")
    original_config = conf.PyngrokConfig

    def configure(**kwargs):
        config = original_config(**kwargs)
        state.configs.append(config)
        return config

    def api_request(url, method="GET", data=None, **kwargs):
        state.requests.append((url, method, dict(data)))
        return {"name": data["name"], "public_url": state.public_url, "url": state.public_url,
                "proto": "https", "config": {"addr": "http://127.0.0.1:8000"},
                "upstream": {"url": "http://127.0.0.1:8000"}}

    monkeypatch.setattr(conf, "PyngrokConfig", configure)
    monkeypatch.setattr(ngrok, "get_ngrok_process", lambda config: SimpleNamespace(api_url="http://127.0.0.1:4040"))
    monkeypatch.setattr(ngrok, "api_request", api_request)
    monkeypatch.setattr(ngrok, "kill", lambda pyngrok_config: state.killed.append(pyngrok_config))
    monkeypatch.setattr(ngrok, "_current_tunnels", {})
    # Paths belong only to this test; never load a user's ngrok configuration.
    monkeypatch.setattr(ngrok.installer, "get_ngrok_config", lambda *a, **k: pytest.fail("No existing config should be read"))
    yield SimpleNamespace(conf=conf, ngrok=ngrok, state=state, configure=configure,
                          logfile=logfile, env_dir=tmp_path / "managed" / "venv")
    studio_logging.close_logging()


@pytest.mark.parametrize("schema", [None, "2"])
def test_real_sdk_builds_valid_upstream_and_https_options(sdk, monkeypatch, schema):
    if schema:
        def configure(**kwargs):
            config = sdk.configure(**kwargs)
            if hasattr(config, "config_version"):
                config.config_version = schema
            return config
        monkeypatch.setattr(sdk.conf, "PyngrokConfig", configure)

    tunnel, close = bootstrap.open_tunnel(8000, sdk.env_dir)
    assert tunnel.public_url == "https://example.test"
    assert len(sdk.state.requests) == 1
    url, method, data = sdk.state.requests[0]
    config = sdk.state.configs[0]
    assert method == "POST"
    assert config.auth_token is None
    assert config.start_new_session is True
    if getattr(config, "config_version", "2") == "3":
        assert urlsplit(url).path == "/api/endpoints"
        assert data["upstream"] == {"url": "http://127.0.0.1:8000"}
        assert data["url"] == "https://"
        assert not {"bind_tls", "schemes", "addr", "proto"} & data.keys()
    else:
        assert urlsplit(url).path == "/api/tunnels"
        assert data["addr"] == "http://127.0.0.1:8000"
        assert data["proto"] == "http" and data["schemes"] == ["https"]
        assert "upstream" not in data and "url" not in data
    assert not sdk.state.killed
    close()
    assert sdk.state.killed == [config]
    saved = sdk.logfile.read_text(encoding="utf-8")
    assert "synthetic-private-token" not in saved and "https://example.test" not in saved
    assert "pyngrok" in saved


def test_legacy_config_without_version_uses_https_tunnel_options(sdk, monkeypatch):
    # pyngrok 7.2 has neither a config_version field nor constructor parameter.
    monkeypatch.setattr(sdk.conf, "PyngrokConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    calls = []
    monkeypatch.setattr(sdk.ngrok, "connect", lambda **kwargs: calls.append(kwargs) or SimpleNamespace(public_url="https://example.test"))
    bootstrap.open_tunnel(8000, sdk.env_dir)
    assert calls[0]["addr"] == "http://127.0.0.1:8000"
    assert calls[0]["bind_tls"] is True and "url" not in calls[0]


def test_non_https_result_is_closed_without_publishing_url(sdk):
    sdk.state.public_url = "http://private-endpoint.test"
    with pytest.raises(RuntimeError, match="could not create an HTTPS endpoint") as raised:
        bootstrap.open_tunnel(8000, sdk.env_dir)
    assert len(sdk.state.killed) == 1
    assert "private-endpoint" not in str(raised.value) + sdk.logfile.read_text(encoding="utf-8")


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_http_rejection_retains_safe_codes_even_when_cleanup_fails(sdk, monkeypatch, cleanup_fails):
    from pyngrok.exception import PyngrokNgrokHTTPError

    private = "synthetic-private-provider-response"
    failure = PyngrokNgrokHTTPError(
        private, "http://private-url.test", 400, private, {"Authorization": private},
        json.dumps({"error_code": 103, "msg": private, "details": {"err": "ERR_NGROK_108 " + private}}),
    )

    def reject(*args, **kwargs):
        raise failure

    def cleanup(pyngrok_config):
        sdk.state.killed.append(pyngrok_config)
        if cleanup_fails:
            raise OSError(private)

    monkeypatch.setattr(sdk.ngrok, "api_request", reject)
    monkeypatch.setattr(sdk.ngrok, "kill", cleanup)
    with pytest.raises(RuntimeError, match="HTTP 400, ERR_NGROK_108, agent error 103") as raised:
        bootstrap.open_tunnel(8000, sdk.env_dir)
    assert len(sdk.state.killed) == 1
    saved = sdk.logfile.read_text(encoding="utf-8")
    events = [json.loads(line) for line in saved.splitlines()]
    failed = next(event for event in events if event["event"] == "tunnel.failed")
    assert failed["error"]["http_status"] == 400
    assert failed["error"]["ngrok_error_code"] == "ERR_NGROK_108"
    assert any(frame["function"] == "reject" for frame in failed["error"]["traceback"])
    assert any(event["event"] == "tunnel.cleanup_failed" for event in events) is cleanup_fails
    assert private not in saved + str(raised.value)
    assert "private-url" not in saved + str(raised.value)
