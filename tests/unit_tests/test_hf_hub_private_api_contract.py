"""Contract test for the private ``huggingface_hub`` internals used by hf_auth.

These back the device-code OAuth flow in ``apps.sources.hf_auth``.

``hf_auth`` reaches into private ``huggingface_hub`` modules because the Hub
exposes device-code login only through its CLI, not through a public Python
API. Those internals are not covered by semver and have moved before (the
``_oauth_device`` module lived at the top level in 1.20.0 and moved to
``utils/`` in 1.20.1, where it has since stayed), so ``huggingface-hub`` is
range-pinned (``>=1.20.1,<2.0.0``) in ``pyproject.toml``.

Unlike ``test_hf_auth_device_code.py`` — which stubs these symbols to test our
orchestration — this test imports the *real* symbols from the *installed* Hub
and checks they are callable the way ``hf_auth`` calls them. Because the pin is
a range, CI runs this test twice: once against the locked version (the normal
``uv sync --frozen`` pytest job) and once against the latest version allowed by
the range (a dedicated job — see .github/workflows/pytest.yml). So an upstream
change to any of these private APIs surfaces here in CI, whether it lands in the
locked version or only in a newer release we have not adopted yet.

When this test fails: re-read the imports in
``src/reachy_mini/apps/sources/hf_auth.py`` against the offending Hub version,
fix them, and (if the floor is affected) update ``_MIN_HF_HUB_VERSION`` below
and the pin in ``pyproject.toml``.
"""

from __future__ import annotations

import inspect

import huggingface_hub
import pytest
from packaging.version import Version

# Keep in sync with the range pin in pyproject.toml ("huggingface-hub>=...,<...").
_MIN_HF_HUB_VERSION = "1.20.1"  # first release with `utils._oauth_device`
_MAX_HF_HUB_VERSION_EXCLUSIVE = "2.0.0"


def test_installed_version_within_supported_range() -> None:
    """Guard the range: the private-API contract is only claimed within it."""
    installed = Version(huggingface_hub.__version__)
    assert (
        Version(_MIN_HF_HUB_VERSION)
        <= installed
        < Version(_MAX_HF_HUB_VERSION_EXCLUSIVE)
    ), (
        f"huggingface_hub {installed} is outside the supported range "
        f"[{_MIN_HF_HUB_VERSION}, {_MAX_HF_HUB_VERSION_EXCLUSIVE}). The private "
        "imports in apps/sources/hf_auth.py are only verified within it; re-check "
        "them against this version and update the pin and this test if adopting it."
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


def test_poll_device_token_accepts_device_info_and_on_pending() -> None:
    """hf_auth calls ``poll_device_token(device_info, on_pending=...)``.

    The ``on_pending`` hook is how ``cancel_device_code_session`` interrupts the
    blocking poll, so its presence is part of the contract we depend on.
    """
    from huggingface_hub.utils._oauth_device import poll_device_token

    inspect.signature(poll_device_token).bind(
        {"device_code": "x", "interval": 5}, on_pending=lambda: None
    )


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
