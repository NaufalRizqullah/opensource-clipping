"""Shared CLI/Studio defaults; vLLM runs in its own server environment."""

import math
import os
from urllib.parse import urlsplit, urlunsplit
import ipaddress

from .base import ProviderConfigError

DEFAULT_VLLM_MODEL = "google/gemma-4-E2B-it-qat-w4a16-ct"
DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8001"
DEFAULT_LOCAL_MAX_TOKENS = 4096


def normalize_base_url(value):
    """Return service root and API base; preserve a reverse proxy path prefix.

    Endpoints are operator-controlled (env/CLI), not per-job web input. Do not
    allow credentials in URLs, metadata/link-local targets, or redirects.
    """
    try:
        parsed = urlsplit(str(value))
        port = parsed.port
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment or port == 0
                or any(c.isspace() or ord(c) < 32 for c in str(value))):
            raise ValueError
        host = parsed.hostname.lower()
        if host in {"metadata.google.internal", "metadata.goog"}:
            raise ValueError
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None:
            address = getattr(address, "ipv4_mapped", None) or address
            if address.is_link_local or address.is_unspecified or address.is_multicast:
                raise ValueError
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        if path.endswith("/v1") or ".." in path.split("/") or "%" in path or "\\" in str(value):
            raise ValueError
        root = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
        return root, root + "/v1"
    except (TypeError, ValueError):
        raise ProviderConfigError("VLLM_BASE_URL harus URL http/https tanpa credentials/query; target metadata/link-local tidak diizinkan.") from None


def _positive(value, name, *, integer=True):
    try:
        if isinstance(value, bool):
            raise ValueError
        result = int(value) if integer else float(value)
        if not math.isfinite(result) or result <= 0:
            raise ValueError
        if integer and float(value) != result:
            raise ValueError
        return result
    except (TypeError, ValueError, OverflowError):
        raise ProviderConfigError(f"{name} harus angka positif{' bulat' if integer else ''}.") from None


def vllm_config_values(values=None, environ=None):
    values = values or {}
    env = os.environ if environ is None else environ

    def pick(field, key, default):
        value = values.get(field)
        return env.get(key, default) if value is None else value

    model = pick("vllm_model", "VLLM_MODEL", DEFAULT_VLLM_MODEL)
    if not isinstance(model, str) or not model.strip() or len(model) > 256 or any(ord(c) < 33 for c in model):
        raise ProviderConfigError("VLLM_MODEL harus nama/alias model served yang valid (maksimum 256 karakter).")
    root, _ = normalize_base_url(pick("vllm_base_url", "VLLM_BASE_URL", DEFAULT_VLLM_BASE_URL))
    context = pick("vllm_context_limit", "VLLM_CONTEXT_LIMIT", None)
    return dict(
        vllm_model=model,
        vllm_base_url=root,
        api_key_vllm=pick("api_key_vllm", "VLLM_API_KEY", ""),
        local_max_tokens=_positive(pick("local_max_tokens", "VLLM_MAX_TOKENS", DEFAULT_LOCAL_MAX_TOKENS), "VLLM_MAX_TOKENS"),
        vllm_context_limit=None if context in (None, "") else _positive(context, "VLLM_CONTEXT_LIMIT"),
        vllm_timeout=_positive(pick("vllm_timeout", "VLLM_TIMEOUT", 900), "VLLM_TIMEOUT", integer=False),
    )
