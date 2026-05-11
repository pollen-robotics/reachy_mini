"""Drift check: the BLE service's inlined hardware-ID helpers must stay
in sync with ``reachy_mini.utils.hardware_id``.

The BLE service is launched as a separate systemd unit running on the
system Python interpreter, outside the ``reachy_mini`` package install.
That's why ``bluetooth_service.py`` reimplements ``_read_raw_audio_serial``
/ ``get_hardware_id`` / ``get_pin`` rather than importing them. If the
two copies drift, the GATT-published hardware ID would diverge from
what the daemon reports over HTTP and WebRTC for the same robot, which
would silently break fleet-management cross-references.

This test parses both files as plain text and asserts that the critical
literals (Pollen audio device VID/PID, SHA-256 truncation length, PIN
fallback) match. It avoids importing ``bluetooth_service`` directly
because that module pulls in ``dbus``/``gi`` which are not part of the
test environment.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
UTILS_PATH = REPO_ROOT / "src" / "reachy_mini" / "utils" / "hardware_id.py"
BLE_PATH = (
    REPO_ROOT
    / "src"
    / "reachy_mini"
    / "daemon"
    / "app"
    / "services"
    / "bluetooth"
    / "bluetooth_service.py"
)


def test_audio_vid_pid_match() -> None:
    """Both files must target the same Pollen audio device (VID 38fb / PID 1001)."""
    utils_src = UTILS_PATH.read_text()
    ble_src = BLE_PATH.read_text()

    utils_vid = re.search(r'POLLEN_AUDIO_VID\s*=\s*"([0-9a-f]+)"', utils_src)
    utils_pid = re.search(r'POLLEN_AUDIO_PID\s*=\s*"([0-9a-f]+)"', utils_src)
    assert utils_vid and utils_pid, "utils/hardware_id.py is missing VID/PID constants"

    vid = utils_vid.group(1)
    pid = utils_pid.group(1)

    # The BLE service inlines the literals as bare strings inside its
    # _read_raw_audio_serial. We only require they appear in the file —
    # the surrounding logic is structurally identical by code review.
    assert f'"{vid}"' in ble_src, (
        f"VID drift: utils declares {vid!r}, not present in bluetooth_service.py"
    )
    assert f'"{pid}"' in ble_src, (
        f"PID drift: utils declares {pid!r}, not present in bluetooth_service.py"
    )


def test_hash_scheme_matches() -> None:
    """Both files must use SHA-256 truncated to the same length."""
    pattern = re.compile(r"hashlib\.sha256\(.*?\)\.hexdigest\(\)\[:(\d+)\]")
    utils_trunc = pattern.findall(UTILS_PATH.read_text())
    ble_trunc = pattern.findall(BLE_PATH.read_text())

    assert utils_trunc, "utils/hardware_id.py: SHA-256 truncation call not found"
    assert ble_trunc, "bluetooth_service.py: SHA-256 truncation call not found"
    assert set(utils_trunc) == set(ble_trunc), (
        f"hash truncation drift: utils={utils_trunc} vs BLE={ble_trunc}"
    )


def test_pin_fallback_matches() -> None:
    """Both files must fall back to the same default PIN when no robot is attached."""
    utils_src = UTILS_PATH.read_text()
    ble_src = BLE_PATH.read_text()

    utils_default = re.search(r'default_pin\s*=\s*"(\d+)"', utils_src)
    assert utils_default, "utils/hardware_id.py: default_pin constant not found"
    default = utils_default.group(1)

    # The BLE copy doesn't introduce a named constant; it just returns the
    # literal in get_pin().
    assert f'"{default}"' in ble_src, (
        f"PIN fallback drift: utils default {default!r} "
        f"not present in bluetooth_service.py"
    )
