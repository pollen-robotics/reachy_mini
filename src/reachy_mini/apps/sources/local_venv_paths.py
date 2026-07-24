"""Stdlib-only venv path/metadata helpers for local common-venv apps.

Extracted from ``local_common_venv.py`` (which re-exports them) so the early
boot path can resolve an app's interpreter, module, and parking support
without importing that module's heavy dependencies (huggingface_hub). Keep
this module free of non-stdlib imports — it runs before the daemon's heavy
imports on the wireless unit.
"""

import platform
import re
import subprocess
import sys
from importlib.metadata import entry_points
from pathlib import Path


def _is_windows() -> bool:
    """Check if the current platform is Windows."""
    return platform.system() == "Windows"


def _should_use_separate_venvs(
    wireless_version: bool = False, desktop_app_daemon: bool = False
) -> bool:
    """Determine if we should use a shared apps_venv (separate from the daemon env)."""
    # Both desktop and wireless use a shared apps_venv for all apps
    return desktop_app_daemon or wireless_version


def _get_venv_parent_dir() -> Path:
    """Get the parent directory of the current venv (OS-agnostic)."""
    # sys.executable is typically: /path/to/venv/bin/python (Linux/Mac)
    # or: C:\path\to\venv\Scripts\python.exe (Windows)
    executable = Path(sys.executable)

    # Determine expected subdirectory based on platform
    expected_subdir = "Scripts" if _is_windows() else "bin"

    # Go up from bin/python or Scripts/python.exe to venv dir, then to parent
    if executable.parent.name == expected_subdir:
        venv_dir = executable.parent.parent
        return venv_dir.parent

    # Fallback: assume we're already in the venv root
    return executable.parent.parent


def _get_app_venv_path(
    app_name: str,
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
) -> Path:
    """Get the venv path for a given app (sibling to current venv).

    Both wireless and desktop use a shared 'apps_venv' for all apps.
    """
    parent_dir = _get_venv_parent_dir()
    return parent_dir / "apps_venv"


def _get_app_python(
    app_name: str,
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
) -> Path:
    """Get the Python executable path for a given app (OS-agnostic)."""
    venv_path = _get_app_venv_path(app_name, wireless_version, desktop_app_daemon)

    if _is_windows():
        # Windows: Scripts/python.exe
        python_exe = venv_path / "Scripts" / "python.exe"
        if python_exe.exists():
            return python_exe
        # Fallback without .exe
        python_path = venv_path / "Scripts" / "python"
        if python_path.exists():
            return python_path
        # Default
        return venv_path / "Scripts" / "python.exe"
    else:
        # Linux/Mac: bin/python
        python_path = venv_path / "bin" / "python"
        if python_path.exists():
            return python_path
        # Default
        return venv_path / "bin" / "python"


def _get_app_site_packages(
    app_name: str,
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
) -> Path | None:
    """Get the site-packages directory for a given app's venv (OS-agnostic)."""
    venv_path = _get_app_venv_path(app_name, wireless_version, desktop_app_daemon)

    if _is_windows():
        # Windows: Lib/site-packages
        site_packages = venv_path / "Lib" / "site-packages"
        if site_packages.exists():
            return site_packages
        return None
    else:
        # Linux/Mac: lib/python3.x/site-packages
        lib_dir = venv_path / "lib"
        if not lib_dir.exists():
            return None
        python_dirs = list(lib_dir.glob("python3.*"))
        if not python_dirs:
            return None
        return python_dirs[0] / "site-packages"


def get_app_site_packages(
    app_name: str,
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
) -> Path | None:
    """Public API to get the site-packages directory for a given app's venv.

    For separate venvs: returns the app's venv site-packages
    For shared environment (SDK mode): returns the current environment's site-packages
    """
    if _should_use_separate_venvs(wireless_version, desktop_app_daemon):
        return _get_app_site_packages(app_name, wireless_version, desktop_app_daemon)
    else:
        # SDK mode: apps are in current environment
        import sysconfig

        return Path(sysconfig.get_paths()["purelib"])


def get_app_python(
    app_name: str,
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
) -> Path:
    """Get the Python executable path for an app (cross-platform).

    For separate venvs: returns the app's venv Python
    For shared environment: returns the current Python interpreter
    """
    if _should_use_separate_venvs(wireless_version, desktop_app_daemon):
        return _get_app_python(app_name, wireless_version, desktop_app_daemon)
    else:
        return Path(sys.executable)


def _find_app_main_file(
    app_name: str,
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
) -> Path | None:
    """Locate an app's ``main.py`` without importing it.

    Tries the app venv's ``site-packages/<app>/main.py`` first (a regular
    copy install, no subprocess). For an **editable** (``-e``) install there is
    no physical ``main.py`` under site-packages (the ``.pth`` redirects imports
    to the source tree), so we resolve the package origin with the app's **own**
    python (``get_app_python`` — the ``apps_venv`` interpreter on wireless /
    desktop, ``sys.executable`` in SDK mode). Doing it out-of-process is what
    makes editable installs work in a *separate* venv: the daemon interpreter
    can't ``find_spec`` an app it can't import. Same subprocess idiom as
    ``check_and_sync_apps_venv_sdk``; bounded and fail-open to ``None``.
    """
    site_packages = _get_app_site_packages(
        app_name, wireless_version, desktop_app_daemon
    )
    if site_packages and site_packages.exists():
        main_file = site_packages / app_name / "main.py"
        if main_file.exists():
            return main_file

    app_python = get_app_python(app_name, wireless_version, desktop_app_daemon)
    try:
        result = subprocess.run(
            [
                str(app_python),
                "-c",
                f"import importlib.util as u; s = u.find_spec({app_name!r}); "
                f"print((s.origin or '') if s else '')",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    origin = result.stdout.strip()
    if origin:
        main_file = Path(origin).parent / "main.py"
        if main_file.exists():
            return main_file
    return None


def app_supports_parking(
    app_name: str,
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
) -> bool:
    """Whether the app opts into parked pre-spawn AND its SDK has the hook.

    Both checks are text scrapes (no import): the app's main.py must declare
    ``supports_parking = True`` and the app venv's reachy_mini must define
    ``_park_until_activated``. Fail-closed: any doubt means no pre-spawn — a
    pre-spawned app that ignored the park flag would boot fully (robot, media,
    wake move) without holding the robot app lock.
    """
    main_file = _find_app_main_file(app_name, wireless_version, desktop_app_daemon)
    if main_file is None:
        return False
    try:
        content = main_file.read_text(encoding="utf-8")
    except OSError:
        return False
    if not re.search(r"supports_parking\s*(?::\s*[^=]+)?\s*=\s*True", content):
        return False

    site_packages = get_app_site_packages(
        app_name, wireless_version, desktop_app_daemon
    )
    if site_packages is None:
        return False
    sdk_app_py = site_packages / "reachy_mini" / "apps" / "app.py"
    try:
        return "_park_until_activated" in sdk_app_py.read_text(encoding="utf-8")
    except OSError:
        # e.g. editable SDK install: skip parking, plain spawn still works.
        return False


def get_app_module(
    app_name: str,
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
) -> str:
    """Get the module name for an app without loading it (for subprocess execution)."""
    if _should_use_separate_venvs(wireless_version, desktop_app_daemon):
        # Get module from separate venv's entry points
        site_packages = _get_app_site_packages(
            app_name, wireless_version, desktop_app_daemon
        )
        if not site_packages or not site_packages.exists():
            raise ValueError(f"App '{app_name}' venv not found or invalid")

        sys.path.insert(0, str(site_packages))
        try:
            eps = entry_points(group="reachy_mini_apps")
            ep = eps.select(name=app_name)
            if not ep:
                raise ValueError(f"No entry point found for app '{app_name}'")
            # Get module name without loading (e.g., "my_app.main" from "my_app.main:MyApp")
            return list(ep)[0].module
        finally:
            sys.path.pop(0)
    else:
        # Get module from current environment
        eps = entry_points(group="reachy_mini_apps", name=app_name)
        ep_list = list(eps)
        if not ep_list:
            raise ValueError(f"No entry point found for app '{app_name}'")
        return ep_list[0].module
