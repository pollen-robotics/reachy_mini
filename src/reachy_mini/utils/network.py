"""Network trust-boundary helpers."""

import ipaddress
from urllib.parse import urlsplit, urlunsplit


def validate_secure_http_url(url: str, setting_name: str) -> str:
    """Validate an HTTP endpoint that may receive bearer credentials."""
    if "\\" in url or any(map(str.isspace, url)):
        raise ValueError(f"{setting_name} must be a valid HTTP(S) URL")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        raise ValueError(f"{setting_name} must be a valid HTTP(S) URL") from None
    if port == 0 or parsed.netloc.endswith(":"):
        raise ValueError(f"{setting_name} must be a valid HTTP(S) URL")

    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError(f"{setting_name} must be an HTTP(S) URL")
    if (
        parsed.username is not None
        or parsed.password is not None
        or "?" in url
        or "#" in url
    ):
        raise ValueError(
            f"{setting_name} must not contain credentials, a query, or a fragment"
        )

    host = parsed.hostname.rstrip(".").lower()
    try:
        is_loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = host == "localhost"
    if parsed.scheme != "https" and not is_loopback:
        raise ValueError(f"{setting_name} must use HTTPS outside loopback")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))
