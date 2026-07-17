"""Drift guard for the hardware-id logic duplicated in the BLE service.

``reachy_mini.utils.hardware_id`` is the canonical implementation, but
``daemon/app/services/bluetooth/bluetooth_service.py`` runs on the SYSTEM
python (see its systemd unit), not the daemon venv, so it CANNOT import the
package (whose ``__init__`` chains into numpy/scipy). It therefore inlines a
copy of the hardware-id / BLE-name / PIN logic.

This test parses both files as plain text - deliberately NOT importing the
dbus/gi-heavy service module - and asserts the shared literals stay in
lock-step, so the two copies can't silently drift apart. It's the drift-check
referenced by the inline-copy note at the top of ``bluetooth_service.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import reachy_mini

_PKG_ROOT = Path(reachy_mini.__file__).resolve().parent
CANONICAL = _PKG_ROOT / "utils" / "hardware_id.py"
INLINED = _PKG_ROOT / "daemon" / "app" / "services" / "bluetooth" / "bluetooth_service.py"


def test_source_files_exist() -> None:
    """Both the canonical and inlined copies must be present."""
    assert CANONICAL.is_file(), CANONICAL
    assert INLINED.is_file(), INLINED


# Literal fragments that MUST appear verbatim in BOTH files. If the canonical
# logic changes, update the inlined copy (and vice-versa) so these keep matching.
SHARED_LITERALS = [
    # Pollen audio device USB ids (the hardware-id source of truth).
    'POLLEN_AUDIO_VID = "38fb"',
    'POLLEN_AUDIO_PID = "1001"',
    # Advertised BLE name prefix.
    'BLE_NAME_PREFIX = "Reachy Mini"',
    # get_hardware_id: SHA-256 of the raw serial, truncated to 16 hex chars.
    'hashlib.sha256(raw.encode("ascii")).hexdigest()[:16]',
    # get_ble_name: prefix + uppercased last-4-hex suffix ("Reachy Mini #XXXX").
    'return f"{BLE_NAME_PREFIX} #{hw[-4:].upper()}"',
    # get_pin: last 5 chars of the raw serial, with a fixed dev-workstation fallback.
    'default_pin = "46879"',
    "return raw[-5:]",
]


@pytest.mark.parametrize("literal", SHARED_LITERALS)
def test_shared_literal_present_in_both(literal: str) -> None:
    """Each shared literal must appear verbatim in both source files."""
    canonical = CANONICAL.read_text(encoding="utf-8")
    inlined = INLINED.read_text(encoding="utf-8")
    assert literal in canonical, f"missing from canonical hardware_id.py: {literal!r}"
    assert literal in inlined, f"missing from inlined bluetooth_service.py: {literal!r}"


@pytest.mark.parametrize("func", ["_read_raw_audio_serial", "get_hardware_id", "get_ble_name", "get_pin"])
def test_shared_function_defined_in_both(func: str) -> None:
    """Both files must define the same set of hardware-id helpers."""
    canonical = CANONICAL.read_text(encoding="utf-8")
    inlined = INLINED.read_text(encoding="utf-8")
    signature = f"def {func}("
    assert signature in canonical, f"{func} missing from canonical hardware_id.py"
    assert signature in inlined, f"{func} missing from inlined bluetooth_service.py"
