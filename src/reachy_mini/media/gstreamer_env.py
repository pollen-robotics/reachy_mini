"""Process-wide environment fixes for GStreamer."""

from __future__ import annotations

import os
import platform


def _dedupe_path_list(value: str) -> list[str]:
    seen: set[str] = set()
    parts: list[str] = []
    for part in value.split(os.pathsep):
        if not part or part in seen:
            continue
        seen.add(part)
        parts.append(part)
    return parts


def configure_gstreamer_environment() -> None:
    """Apply macOS-specific GStreamer environment fixes early.

    These changes must be safe to call before importing ``gi`` / ``Gst``.
    """
    if platform.system() != "Darwin":
        return

    os.environ.setdefault("GST_REGISTRY_FORK", "no")

    # ``gstreamer_python`` exposes ``libgstpython.dylib`` in the plugin search
    # path. On some uv/venv setups that library links against python.org's
    # framework build instead of the active interpreter and can segfault during
    # plugin scanning. Reachy Mini does not need that plugin at runtime.
    plugin_dir_fragment = "gstreamer_python/lib/gstreamer-1.0"
    for key in ("GST_PLUGIN_PATH_1_0", "GST_PLUGIN_SYSTEM_PATH_1_0"):
        value = os.environ.get(key)
        if not value:
            continue

        parts = [
            part
            for part in _dedupe_path_list(value)
            if part and plugin_dir_fragment not in part
        ]
        if parts:
            os.environ[key] = os.pathsep.join(parts)
        else:
            os.environ.pop(key, None)

    # These are single-value vars. Some environments prepend the same value to
    # itself during startup, producing malformed "path:path" strings that
    # GStreamer does not consistently tolerate.
    for key in ("GST_REGISTRY_1_0", "GST_PLUGIN_SCANNER_1_0", "GST_PYTHONPATH_1_0"):
        value = os.environ.get(key)
        if not value:
            continue

        parts = _dedupe_path_list(value)
        if parts:
            os.environ[key] = parts[0]
        else:
            os.environ.pop(key, None)
