"""Contract test for the private ``huggingface_hub`` internals used by hf_auth.

These back the device-code OAuth flow in ``apps.sources.hf_auth``.

``hf_auth`` reaches into private ``huggingface_hub`` modules because the Hub
exposes device-code login only through its CLI, not through a public Python
API. Those internals are not covered by semver and have already moved between
releases (the ``_oauth_device`` module lived at the top level in 1.20.0 and
moved to ``utils/`` in 1.20.1), so ``huggingface-hub`` is pinned to an exact
version in ``pyproject.toml``.

Unlike ``test_hf_auth_device_code.py`` — which stubs these symbols to test our
orchestration — this test imports the *real* symbols from the *installed* Hub
and checks they are callable the way ``hf_auth`` calls them. It runs in CI
against the pinned version, so bumping ``huggingface-hub`` without re-verifying
the private imports fails here loudly instead of at runtime on a robot.

When this test fails after a version bump: re-read the imports in
``src/reachy_mini/apps/sources/hf_auth.py`` against the new Hub source, fix
them, then update ``_PINNED_HF_HUB_VERSION`` below and the pin in
``pyproject.toml``.
"""

from __future__ import annotations

import inspect

import huggingface_hub
import pytest

# Keep in sync with the exact pin in pyproject.toml ("huggingface-hub==...").
_PINNED_HF_HUB_VERSION = "1.20.1"


def test_installed_version_matches_pin() -> None:
    """Guard the pin: a bump must be a deliberate edit that revisits this test."""
    assert huggingface_hub.__version__ == _PINNED_HF_HUB_VERSION, (
        f"huggingface_hub {huggingface_hub.__version__} installed but "
        f"{_PINNED_HF_HUB_VERSION} is pinned. If this is an intentional bump, "
        "re-verify every private import in apps/sources/hf_auth.py against the "
        "new version, then update the pin and _PINNED_HF_HUB_VERSION."
    )


def test_device_code_protocol_imports_exist() -> None:
    """The RFC 8628 protocol helpers hf_auth imports must be importable."""
    from huggingface_hub.utils._oauth_device import (  # noqa: F401
        poll_device_token,
        request_device_code,
    )


def test_request_device_code_callable_with_no_args() -> None:
    """hf_auth calls ``request_device_code()`` — no arguments."""
    from huggingface_hub.utils._oauth_device import request_device_code

    # bind() (not the network call) verifies the call shape hf_auth relies on.
    inspect.signature(request_device_code).bind()


def test_poll_device_token_accepts_device_info() -> None:
    """hf_auth calls ``poll_device_token(device_info)`` — one positional arg."""
    from huggingface_hub.utils._oauth_device import poll_device_token

    inspect.signature(poll_device_token).bind({"device_code": "x", "interval": 5})


def test_save_oauth_token_accepts_response_and_is_private() -> None:
    """hf_auth calls ``_save_oauth_token(response)`` and unpacks (name, user)."""
    from huggingface_hub._login import _save_oauth_token

    inspect.signature(_save_oauth_token).bind({"access_token": "x"})


def test_device_code_error_is_exception() -> None:
    """hf_auth catches ``DeviceCodeError`` from the polling loop."""
    from huggingface_hub.errors import DeviceCodeError

    assert issubclass(DeviceCodeError, Exception)


@pytest.mark.parametrize(
    "module_path, attr",
    [
        ("huggingface_hub.utils._oauth_device", "request_device_code"),
        ("huggingface_hub.utils._oauth_device", "poll_device_token"),
        ("huggingface_hub._login", "_save_oauth_token"),
        ("huggingface_hub.errors", "DeviceCodeError"),
    ],
)
def test_private_symbols_are_callable_or_class(module_path: str, attr: str) -> None:
    """Every private symbol hf_auth depends on resolves and is usable."""
    import importlib

    module = importlib.import_module(module_path)
    symbol = getattr(module, attr)
    assert inspect.isfunction(symbol) or inspect.isclass(symbol)
