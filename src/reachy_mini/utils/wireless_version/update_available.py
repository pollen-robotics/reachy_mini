"""Check if an update is available for Reachy Mini Wireless.

For now, this only checks if a new version of "reachy_mini" is available on PyPI.
"""

from importlib.metadata import version

import requests
import semver


def is_update_available(package_name: str) -> bool:
    """Check if an update is available for the given package."""
    pypi_version = get_pypi_version(package_name)
    local_version = get_local_version(package_name)
    if semver.compare(pypi_version, local_version) > 0:
        return True
    return False


def get_pypi_version(package_name: str) -> str:
    """Get the latest version of a package from PyPI."""
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data["info"]["version"]


def get_local_version(package_name: str) -> str:
    """Get the currently installed version of a package."""
    return version(package_name)
