"""Process-wide environment fixes for GStreamer."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
from pathlib import Path


def _dedupe_path_list(value: str) -> list[str]:
    seen: set[str] = set()
    parts: list[str] = []
    for part in value.split(os.pathsep):
        if not part or part in seen:
            continue
        seen.add(part)
        parts.append(part)
    return parts


def _prepend_path_list(key: str, new_part: str) -> None:
    """Prepend a single path entry to an environment path list."""
    parts = [new_part]
    parts.extend(
        part
        for part in _dedupe_path_list(os.environ.get(key, ""))
        if part != new_part
    )
    os.environ[key] = os.pathsep.join(parts)


def _iter_plugin_dirs() -> list[Path]:
    """Return the known GStreamer plugin directories from env and bundle metadata."""
    plugin_dirs: list[Path] = []
    seen: set[Path] = set()

    for key in ("GST_PLUGIN_PATH_1_0", "GST_PLUGIN_SYSTEM_PATH_1_0"):
        value = os.environ.get(key, "")
        for part in _dedupe_path_list(value):
            path = Path(part)
            if path in seen:
                continue
            seen.add(path)
            plugin_dirs.append(path)

    spec = importlib.util.find_spec("gstreamer_plugins")
    if spec is not None and spec.submodule_search_locations is not None:
        for location in spec.submodule_search_locations:
            path = Path(location) / "lib" / "gstreamer-1.0"
            if path in seen:
                continue
            seen.add(path)
            plugin_dirs.append(path)

    return plugin_dirs


def _find_disabled_applemedia_plugin() -> Path | None:
    """Find the disabled macOS Apple media plugin shipped by gstreamer-bundle."""
    for plugin_dir in _iter_plugin_dirs():
        candidate = plugin_dir / "libgstapplemedia.dylib.disabled"
        if candidate.is_file():
            return candidate
    return None


def _get_cache_root() -> Path:
    """Return the cache directory used for Reachy Mini's GStreamer fixes."""
    override = os.environ.get("REACHY_MINI_GSTREAMER_CACHE_DIR")
    if override:
        return Path(override)

    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Caches" / "reachy_mini" / "gstreamer"
    return Path.home() / ".cache" / "reachy_mini" / "gstreamer"


def _enable_macos_applemedia_plugin() -> None:
    """Expose `libgstapplemedia` to GStreamer when the wheel ships it disabled.

    The plugin depends on sibling dylibs via relative loader paths, so the
    enabled filename must live in the original plugin directory. Copying it to a
    cache directory breaks those references on macOS.
    """
    disabled_plugin = _find_disabled_applemedia_plugin()
    if disabled_plugin is None:
        return

    cache_root = _get_cache_root()
    enabled_plugin = disabled_plugin.with_suffix("")
    disabled_stat = disabled_plugin.stat()
    if (
        not enabled_plugin.exists()
        or enabled_plugin.stat().st_size != disabled_stat.st_size
        or enabled_plugin.stat().st_mtime_ns < disabled_stat.st_mtime_ns
    ):
        shutil.copy2(disabled_plugin, enabled_plugin)

    for key in ("GST_PLUGIN_PATH_1_0", "GST_PLUGIN_SYSTEM_PATH_1_0"):
        _prepend_path_list(key, str(disabled_plugin.parent))

    # Use a dedicated registry so GStreamer rescans the newly exposed plugin
    # instead of reusing a cache created before `applemedia` was enabled.
    registry_path = cache_root / "registry.bin"
    registry_path.unlink(missing_ok=True)
    os.environ["GST_REGISTRY_1_0"] = str(registry_path)


def configure_gstreamer_environment() -> None:
    """Apply macOS-specific GStreamer environment fixes early.

    These changes must be safe to call before importing ``gi`` / ``Gst``.
    """
    if platform.system() != "Darwin":
        return

    os.environ.setdefault("GST_REGISTRY_FORK", "no")
    _enable_macos_applemedia_plugin()

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
