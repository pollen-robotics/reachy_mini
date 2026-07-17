"""Shared guard for tests that need the gst-plugins-rs webrtc plugin.

The ``webrtc``-marked tests need webrtcsrc/webrtcsink from ``libgstrswebrtc.so``.
They skip cleanly where the plugin isn't installed (local dev, macOS), but a CI
job that set the plugin up exports ``REACHY_MINI_REQUIRE_WEBRTC=1`` so a missing
element fails loudly there instead of silently skipping and dropping coverage.
"""

import os

import gi
import pytest

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402


def require_webrtc_plugin(*elements: str) -> None:
    """Skip — or, when the job requires it, fail — unless every element registers."""
    Gst.init([])
    missing = [e for e in elements if Gst.ElementFactory.find(e) is None]
    if not missing:
        return

    reason = (
        "gst-plugins-rs webrtc plugin unavailable — missing element(s): "
        f"{', '.join(missing)} (libgstrswebrtc.so not loaded)"
    )
    if os.environ.get("REACHY_MINI_REQUIRE_WEBRTC") == "1":
        # This job declared the plugin present; a skip here would hide a broken
        # plugin and silently drop coverage, so fail loudly instead.
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason, allow_module_level=True)
