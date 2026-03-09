"""Utilities for checking app updates from HuggingFace Spaces."""

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path

import aiohttp

from .. import AppInfo

HF_SPACES_API_URL = "https://huggingface.co/api/spaces"
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=10)

# Pattern to extract space ID and SHA from HuggingFace cache path
# e.g., .../spaces--owner--repo/snapshots/3f47979df625a013429913f5bc025f00ca5b92c0/...
HF_CACHE_PATTERN = re.compile(
    r"spaces--([^/]+)--([^/]+)/snapshots/([a-f0-9]{40})(?:/|$)"
)

# Pattern to extract space ID from a HuggingFace spaces URL (git+https installs)
# e.g., https://huggingface.co/spaces/RemiFabre/marionette
HF_SPACES_URL_PATTERN = re.compile(r"huggingface\.co/spaces/([^/]+)/([^/?#]+)")


@dataclass
class AppUpdateInfo:
    """Information about an app's update status."""

    app_name: str
    space_id: str
    installed_sha: str
    latest_sha: str
    update_available: bool
    last_modified: str | None = None


@dataclass
class HfInstallInfo:
    """Information extracted from a HuggingFace-installed package."""

    space_id: str  # e.g., "pollen-robotics/hand_tracker_v2"
    installed_sha: str  # 40-char commit SHA


def _extract_hf_info_from_site_packages(
    site_packages_path: str | Path, app_name: str
) -> HfInstallInfo | None:
    """Search for HuggingFace install info in dist-info within a site-packages directory."""
    site_packages = Path(site_packages_path)
    if not site_packages.exists():
        return None

    # Try different name variations (underscore vs dash, case)
    name_variants = [
        app_name,
        app_name.replace("_", "-"),
        app_name.replace("-", "_"),
        app_name.lower(),
        app_name.lower().replace("_", "-"),
        app_name.lower().replace("-", "_"),
    ]
    seen = set()

    for name in name_variants:
        if name in seen:
            continue
        seen.add(name)

        for dist_info in site_packages.glob(f"{name}*.dist-info"):
            direct_url_path = dist_info / "direct_url.json"
            if direct_url_path.exists():
                try:
                    direct_url = json.loads(direct_url_path.read_text())
                    url = direct_url.get("url", "")

                    # Method 1: Installed from HF cache (snapshot_download + pip install)
                    match = HF_CACHE_PATTERN.search(url)
                    if match:
                        owner, repo, sha = match.groups()
                        return HfInstallInfo(
                            space_id=f"{owner}/{repo}",
                            installed_sha=sha,
                        )

                    # Method 2: Installed via git+https (vcs_info has commit_id)
                    vcs_info = direct_url.get("vcs_info")
                    if vcs_info:
                        commit_id = vcs_info.get("commit_id")
                        url_match = HF_SPACES_URL_PATTERN.search(url)
                        if commit_id and url_match:
                            owner, repo = url_match.groups()
                            return HfInstallInfo(
                                space_id=f"{owner}/{repo}",
                                installed_sha=commit_id,
                            )
                except Exception:
                    pass

    return None


def get_hf_install_info(
    app_name: str,
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
) -> HfInstallInfo | None:
    """Extract HuggingFace install info from a package's direct_url.json.

    This extracts both the space ID and installed commit SHA from pip metadata,
    without requiring any separately stored metadata.

    Args:
        app_name: The app name.
        wireless_version: Whether running on wireless version.
        desktop_app_daemon: Whether running as desktop app daemon.

    Returns:
        HfInstallInfo with space_id and installed_sha, or None if not found.

    """
    from . import local_common_venv

    # Use the existing system to get the correct site-packages path
    site_packages = local_common_venv.get_app_site_packages(
        app_name, wireless_version, desktop_app_daemon
    )
    if site_packages:
        return _extract_hf_info_from_site_packages(site_packages, app_name)

    return None


async def get_space_latest_sha(
    space_id: str, token: str | None = None
) -> tuple[str, str | None] | None:
    """Get the latest SHA and lastModified for a HuggingFace Space.

    Args:
        space_id: The HuggingFace space ID (e.g., "owner/app-name").
        token: Optional HuggingFace token for private spaces.

    Returns:
        Tuple of (sha, lastModified) or None if request fails.

    """
    url = f"{HF_SPACES_API_URL}/{space_id}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("sha"), data.get("lastModified")
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass
    return None


async def check_app_update(
    app: AppInfo,
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
) -> AppUpdateInfo | None:
    """Check if an update is available for a single app.

    Args:
        app: The installed app to check.
        wireless_version: Whether running on wireless version.
        desktop_app_daemon: Whether running as desktop app daemon.

    Returns:
        AppUpdateInfo if check succeeded, None if unable to check.

    """
    # Try to get install info from pip metadata (most reliable, works without stored metadata)
    hf_info = get_hf_install_info(app.name, wireless_version, desktop_app_daemon)

    space_id: str | None = None
    installed_sha: str | None = None
    if hf_info:
        space_id = hf_info.space_id
        installed_sha = hf_info.installed_sha
    else:
        # Fall back to stored metadata
        space_id = app.extra.get("id")
        installed_sha = app.extra.get("installed_sha")

    if not space_id or not installed_sha:
        # Can't determine source or installed version
        return None

    # Get token for private spaces
    token = None
    if app.extra.get("private"):
        from . import hf_auth

        token = hf_auth.get_hf_token()

    # Fetch latest SHA from HuggingFace
    result = await get_space_latest_sha(space_id, token)
    if result is None:
        return None

    latest_sha, last_modified = result

    return AppUpdateInfo(
        app_name=app.name,
        space_id=space_id,
        installed_sha=installed_sha,
        latest_sha=latest_sha,
        update_available=(installed_sha != latest_sha),
        last_modified=last_modified,
    )


async def check_all_app_updates(
    apps: list[AppInfo],
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
    concurrency: int = 3,
    delay_between_batches: float = 1.0,
) -> list[AppUpdateInfo]:
    """Check updates for multiple apps with rate limiting.

    Args:
        apps: List of installed apps to check.
        wireless_version: Whether running on wireless version.
        desktop_app_daemon: Whether running as desktop app daemon.
        concurrency: Max concurrent requests per batch.
        delay_between_batches: Seconds to wait between batches (for "slow" checking).

    Returns:
        List of AppUpdateInfo for apps that could be checked (includes apps with
        and without updates).

    """
    results: list[AppUpdateInfo] = []

    # Process in batches
    for i in range(0, len(apps), concurrency):
        batch = apps[i : i + concurrency]
        batch_results = await asyncio.gather(
            *[
                check_app_update(app, wireless_version, desktop_app_daemon)
                for app in batch
            ],
            return_exceptions=True,
        )

        for result in batch_results:
            if isinstance(result, AppUpdateInfo):
                results.append(result)
            # Skip exceptions and None results (apps that couldn't be checked)

        # Rate limiting between batches
        if i + concurrency < len(apps):
            await asyncio.sleep(delay_between_batches)

    return results
