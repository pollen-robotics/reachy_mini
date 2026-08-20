"""Check if an update is available for Reachy Mini Wireless.

For now, this only checks if a new version of "reachy_mini" is available on PyPI.
"""

import json
import re
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
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    data = response.json()

    version = _semver_version(data["info"]["version"])

    if pre_release:
        # PyPI orders ``releases`` lexicographically, so the last key is not the
        # newest version ("1.9.0rc1" sorts after "1.10.0rc5"). Take the max.
        candidates = [version]
        for v, files in data["releases"].items():
            if not files or all(f.get("yanked") for f in files):
                continue  # deleted or yanked: pip would refuse to install it
            try:
                candidates.append(_semver_version(v))
            except ValueError:
                continue  # version scheme we can't compare (e.g. ``.post1``)
        return max(candidates)

    return version


def get_local_version(package_name: str) -> semver.Version:
    """Get the currently installed version of a package."""
    return _semver_version(version(package_name))


# PEP 440 pre-release / dev markers that sort *before* the release, matching
# semver pre-release ordering. `.postN` is intentionally excluded: it sorts
# after the release, which a semver pre-release token cannot represent.
_PEP440_SUFFIX = re.compile(
    r"^(?P<base>\d+\.\d+\.\d+)\.?(?P<kind>a|b|c|rc|alpha|beta|dev)\.?(?P<num>\d+)$"
)


def _semver_version(v: str) -> semver.Version:
    """Convert a version string to a semver.Version object, handling pypi pre-release formats.

    ``semver.Version.parse`` only accepts strict semver, so PEP 440 suffixes
    such as ``1.2.3rc4`` (attached) and ``1.10.0.dev0`` (dot-separated) are
    normalized to a semver pre-release (``1.2.3-rc.4`` / ``1.10.0-dev.0``)
    first. The package ships ``X.Y.Z.dev0`` on main, so this path is hit by
    ``get_local_version`` on any from-source / from-git-ref install.
    """
    try:
        return semver.Version.parse(v)
    except ValueError:
        pass

    m = _PEP440_SUFFIX.match(v)
    if m is None:
        raise ValueError(f"Invalid version string: {v}")

    return semver.Version.parse(f"{m['base']}-{m['kind']}.{m['num']}")
