"""Tests for the macOS GStreamer environment bootstrap."""

from __future__ import annotations

import os
from pathlib import Path

from reachy_mini.media import gstreamer_env


def test_configure_gstreamer_environment_enables_disabled_applemedia_plugin(
    monkeypatch, tmp_path: Path
) -> None:
    """Expose the disabled applemedia plugin through its original plugin dir."""

    disabled_plugin = tmp_path / "bundle" / "libgstapplemedia.dylib.disabled"
    disabled_plugin.parent.mkdir(parents=True)
    disabled_plugin.write_bytes(b"applemedia")

    cache_root = tmp_path / "cache"

    monkeypatch.setattr(gstreamer_env.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        gstreamer_env,
        "_find_disabled_applemedia_plugin",
        lambda: disabled_plugin,
    )
    monkeypatch.setattr(gstreamer_env, "_get_cache_root", lambda: cache_root)
    monkeypatch.setenv(
        "GST_PLUGIN_PATH_1_0",
        os.pathsep.join(
            [
                str(tmp_path / "python_plugins" / "gstreamer_python/lib/gstreamer-1.0"),
                str(tmp_path / "existing_plugins"),
            ]
        ),
    )
    monkeypatch.delenv("GST_PLUGIN_SYSTEM_PATH_1_0", raising=False)
    monkeypatch.delenv("GST_REGISTRY_1_0", raising=False)

    gstreamer_env.configure_gstreamer_environment()

    enabled_plugin = tmp_path / "bundle" / "libgstapplemedia.dylib"
    assert enabled_plugin.read_bytes() == b"applemedia"
    assert os.environ["GST_PLUGIN_PATH_1_0"].split(os.pathsep) == [
        str(disabled_plugin.parent),
        str(tmp_path / "existing_plugins"),
    ]
    assert os.environ["GST_PLUGIN_SYSTEM_PATH_1_0"] == str(disabled_plugin.parent)
    assert os.environ["GST_REGISTRY_1_0"] == str(cache_root / "registry.bin")
