"""Unit tests for the ReachyMiniOS reflash helpers."""

import asyncio
from unittest.mock import MagicMock

import pytest

from reachy_mini.utils.wireless_version import os_update

RELEASE = {
    "tag_name": "v1.1.1",
    "assets": [
        {"name": "notes.txt", "browser_download_url": "https://x/notes.txt"},
        {"name": "image-v1.1.1.zip", "browser_download_url": "https://x/image.zip"},
        {
            "name": "image-v1.1.1.zip.sha256",
            "browser_download_url": "https://x/image.zip.sha256",
        },
    ],
}


def _response(status=200, json=None, text=""):
    resp = MagicMock(status_code=status, text=text)
    resp.json.return_value = json
    return resp


def test_release_with_sha256(monkeypatch):
    """The image and its published checksum are found."""
    calls = []

    def get(url, timeout):
        calls.append(url)
        if url.endswith(".sha256"):
            return _response(text="abc123  image-v1.1.1.zip\n")
        return _response(json=RELEASE)

    monkeypatch.setattr(os_update.requests, "get", get)

    release = os_update.get_os_release("v1.1.1")

    assert calls[0].endswith(
        "/repos/pollen-robotics/reachy-mini-os/releases/tags/v1.1.1"
    )
    assert release == {
        "tag": "v1.1.1",
        "zip_url": "https://x/image.zip",
        "sha256": "abc123",
    }


def test_latest_release_without_sha256(monkeypatch):
    """No tag means the latest release; a missing checksum is None."""
    release = {**RELEASE, "assets": RELEASE["assets"][:2]}
    get = MagicMock(return_value=_response(json=release))
    monkeypatch.setattr(os_update.requests, "get", get)

    assert os_update.get_os_release()["sha256"] is None
    assert get.call_args[0][0].endswith("/releases/latest")


def test_release_not_found(monkeypatch):
    """A missing release raises OSReleaseError."""
    monkeypatch.setattr(os_update.requests, "get", lambda url, timeout: _response(404))

    with pytest.raises(os_update.OSReleaseError):
        os_update.get_os_release("v9")


def test_release_without_image(monkeypatch):
    """A release with no .zip raises OSReleaseError."""
    release = {**RELEASE, "assets": RELEASE["assets"][:1]}
    monkeypatch.setattr(
        os_update.requests, "get", lambda url, timeout: _response(json=release)
    )

    with pytest.raises(os_update.OSReleaseError):
        os_update.get_os_release()


def test_reflash_runs_script_as_root(monkeypatch):
    """The upgrade script gets the image URL, checksum and dry-run flag."""
    monkeypatch.setattr(
        os_update,
        "get_os_release",
        lambda tag: {
            "tag": "v1.1.1",
            "zip_url": "https://x/image.zip",
            "sha256": "abc",
        },
    )
    commands = []

    async def wrapper(cmd, logger, ok_returncodes=(0,)):
        commands.append(cmd)

    monkeypatch.setattr(os_update, "call_logger_wrapper", wrapper)

    asyncio.run(os_update.reflash_os(MagicMock(), "v1.1.1", dry_run=True))

    assert commands[0].startswith("sudo -n /usr/bin/bash ")
    assert commands[0].endswith(
        "reachy-os-upgrade.sh --yes --dry-run --sha256 abc https://x/image.zip"
    )
