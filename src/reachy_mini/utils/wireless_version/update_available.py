"""Check if an update is available for Reachy Mini Wireless.

For now, this only checks if a new version of "reachy_mini" is available on PyPI.
"""

from importlib.metadata import version

import requests
import semver


def is_update_available(package_name: str, pre_release: bool) -> bool:
    """Check if an update is available for the given package."""
    pypi_version = get_pypi_version(package_name, pre_release)
    local_version = get_local_version(package_name)
    return pypi_version > local_version


def get_pypi_version(package_name: str, pre_release: bool) -> semver.Version:
    """Get the latest version of a package from PyPI."""
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    if pre_release:
        releases = list(data["releases"].keys())
        version = releases[-1]

    else:
        version = data["info"]["version"]

    return _semver_version(version)


def get_local_version(package_name: str) -> semver.Version:
    """Get the currently installed version of a package."""
    return _semver_version(version(package_name))


def _semver_version(v: str) -> semver.Version:
    """Convert a version string to a semver.Version object, handling pypi pre-release formats."""
    try:
        return semver.Version.parse(v)
    except ValueError:
        version_parts = v.split(".")
        if len(version_parts) < 3:
            raise ValueError(f"Invalid version string: {v}")

        patch_part = version_parts[2]
        if "rc" in patch_part:
            patch, rc = patch_part.split("rc", 1)
            v_clean = f"{version_parts[0]}.{version_parts[1]}.{patch}-rc.{rc}"
            return semver.Version.parse(v_clean)

    raise ValueError(f"Invalid version string: {v}")
