"""HTTP endpoint validation without optional media dependencies."""

import pytest

from reachy_mini.utils.network import validate_secure_http_url


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("http://central.example", "HTTPS outside loopback"),
        ("ftp://central.example", "must be an HTTP"),
        ("https:///base", "must be an HTTP"),
        ("https://user:password@central.example", "must not contain"),
        ("https://central.example?tenant=pollen", "must not contain"),
        ("https://central.example#fragment", "must not contain"),
        ("https://central.example?", "must not contain"),
        ("https://central.example#", "must not contain"),
        ("https://central.example:invalid", "valid HTTP"),
        ("https://central.example:0", "valid HTTP"),
        ("https://central.example:65536", "valid HTTP"),
        ("https://central.example:", "valid HTTP"),
        ("https://central.example\\evil.example", "valid HTTP"),
        ("https://central example", "valid HTTP"),
        (" https://central.example", "valid HTTP"),
        ("https://user:secret-marker@exam／ple.com", "valid HTTP"),
    ],
)
def test_rejects_unsafe_urls_without_echoing_credentials(
    url: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message) as error:
        validate_secure_http_url(url, "endpoint")
    assert "secret-marker" not in str(error.value)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/",
        "http://LOCALHOST.:8000/central/",
        "http://127.0.0.2:8000/central/",
        "http://[::1]:8000/central/",
        "https://central.example/base/",
        "https://turn.example/credentials/",
        "https://turn.example/credentials",
    ],
)
def test_preserves_secure_and_loopback_endpoint_paths(url: str) -> None:
    assert validate_secure_http_url(url, "endpoint") == url
