"""Check if an update is available for Reachy Mini Wireless.

For now, this only checks if a new version of "reachy_mini" is available on PyPI.
"""

import json
from importlib.metadata import distribution, version

import requests
import semver


def get_install_source(package_name: str) -> dict[str, str]:
    """Get install source info: version and origin (PyPI, git ref, or editable)."""
    dist = distribution(package_name)
    result = {"version": version(package_name), "source": "pypi"}

    try:
        direct_url_text = dist.read_text("direct_url.json")
        if direct_url_text is None:
            return result
        direct_url = json.loads(direct_url_text)
        if "dir_info" in direct_url and direct_url["dir_info"].get("editable"):
            result["source"] = "editable"
        elif "vcs_info" in direct_url:
            vcs = direct_url["vcs_info"]
            result["source"] = "git"
            result["git_ref"] = vcs.get("requested_revision", "unknown")
            result["commit"] = vcs.get("commit_id", "")[:8]  # Short hash
    except FileNotFoundError:
        pass  # No direct_url.json means PyPI install

    return result


def is_update_available(package_name: str, pre_release: bool) -> bool:
    """Check if an update is available for the given package."""
    pypi_version = get_pypi_version(package_name, pre_release)
    local_version = get_local_version(package_name)

    is_update_available = pypi_version > local_version
    assert isinstance(is_update_available, bool)

    return is_update_available


def get_pypi_version(package_name: str, pre_release: bool) -> semver.Version:
    """Get the latest version of a package from PyPI."""
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    version = data["info"]["version"]

    if pre_release:
        releases = list(data["releases"].keys())
        pre_version = _semver_version(releases[-1])
        if pre_version > version:
            return pre_version

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
