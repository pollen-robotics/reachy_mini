"""Process-wide environment fixes for GStreamer."""

from __future__ import annotations

import importlib.util
import logging
import os
import platform
import shutil
from pathlib import Path

_PLUGIN_PATH_KEYS = ("GST_PLUGIN_PATH_1_0", "GST_PLUGIN_SYSTEM_PATH_1_0")
_SINGLE_PATH_KEYS = (
    "GST_REGISTRY_1_0",
    "GST_PLUGIN_SCANNER_1_0",
    "GST_PYTHONPATH_1_0",
)
_logger = logging.getLogger(__name__)


def _dedupe_path_list(value: str) -> list[str]:
    seen: set[str] = set()
    parts: list[str] = []
    for part in value.split(os.pathsep):
        if not part or part in seen:
            continue
        seen.add(part)
        parts.append(part)
    return parts


def _find_disabled_applemedia_plugin() -> Path | None:
    """Find the disabled macOS Apple media plugin shipped by gstreamer-bundle."""
    spec = importlib.util.find_spec("gstreamer_plugins")
    if spec is not None and spec.submodule_search_locations is not None:
        for location in spec.submodule_search_locations:
            candidate = (
                Path(location)
                / "lib"
                / "gstreamer-1.0"
                / "libgstapplemedia.dylib.disabled"
            )
            if candidate.is_file():
                return candidate
    return None


def configure_gstreamer_environment() -> None:
    """Apply macOS-specific GStreamer environment fixes early.

    These changes must be safe to call before importing ``gi`` / ``Gst``.
    """
    if platform.system() != "Darwin":
        return

    os.environ.setdefault("GST_REGISTRY_FORK", "no")

    disabled_plugin = _find_disabled_applemedia_plugin()
    if disabled_plugin is not None:
        plugin_changes: list[str] = []

        # The plugin depends on sibling dylibs via relative loader paths, so the
        # enabled filename must live in the original plugin directory.
        enabled_plugin = disabled_plugin.with_suffix("")
        disabled_stat = disabled_plugin.stat()
        if (
            not enabled_plugin.exists()
            or enabled_plugin.stat().st_size != disabled_stat.st_size
            or enabled_plugin.stat().st_mtime_ns < disabled_stat.st_mtime_ns
        ):
            shutil.copy2(disabled_plugin, enabled_plugin)
            plugin_changes.append(f"enabled {enabled_plugin.name}")

        plugin_dir = str(disabled_plugin.parent)
        for key in _PLUGIN_PATH_KEYS:
            previous = os.environ.get(key)
            parts = _dedupe_path_list(
                os.pathsep.join([plugin_dir, os.environ.get(key, "")])
            )
            if parts:
                updated = os.pathsep.join(parts)
                os.environ[key] = updated
                if updated != previous:
                    plugin_changes.append(f"updated {key}")
            else:
                os.environ.pop(key, None)
                if previous is not None:
                    plugin_changes.append(f"cleared {key}")

        # Use a dedicated registry so GStreamer rescans the newly exposed
        # plugin instead of reusing a cache created before `applemedia`
        # was enabled.
        registry_dir = Path(
            os.environ.get(
                "REACHY_MINI_GSTREAMER_CACHE_DIR",
                Path.home() / "Library" / "Caches" / "reachy_mini" / "gstreamer",
            )
        )
        registry_dir.mkdir(parents=True, exist_ok=True)
        registry_path = registry_dir / "registry.bin"
        registry_existed = registry_path.exists()
        registry_path.unlink(missing_ok=True)
        previous_registry = os.environ.get("GST_REGISTRY_1_0")
        os.environ["GST_REGISTRY_1_0"] = str(registry_path)
        if registry_existed or previous_registry != str(registry_path):
            plugin_changes.append("reset GST_REGISTRY_1_0")

        if plugin_changes:
            _logger.info(
                "Applied macOS GStreamer applemedia workaround: %s",
                ", ".join(plugin_changes),
            )

    # ``gstreamer_python`` exposes ``libgstpython.dylib`` in the plugin search
    # path. On some uv/venv setups that library links against python.org's
    # framework build instead of the active interpreter and can segfault during
    # plugin scanning. Reachy Mini does not need that plugin at runtime.
    plugin_dir_fragment = "gstreamer_python/lib/gstreamer-1.0"
    for key in _PLUGIN_PATH_KEYS:
        parts = [
            part
            for part in _dedupe_path_list(os.environ.get(key, ""))
            if plugin_dir_fragment not in part
        ]
        if parts:
            os.environ[key] = os.pathsep.join(parts)
        else:
            os.environ.pop(key, None)

    # These are single-value vars. Some environments prepend the same value to
    # itself during startup, producing malformed "path:path" strings that
    # GStreamer does not consistently tolerate.
    for key in _SINGLE_PATH_KEYS:
        value = os.environ.get(key)
        if not value:
            continue

        parts = _dedupe_path_list(value)
        if parts:
            os.environ[key] = parts[0]
        else:
            os.environ.pop(key, None)
