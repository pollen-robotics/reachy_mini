"""Serial port discovery for the NFC reader board.

The NFC reader (the Winnie board, a CLRC663 driven by ``winnie_nfc``) bridges
USB with a CH343, which exposes **exactly the same USB vendor and product ids
as the Reachy Mini motor controller** — ``1a86:55d3`` for both. USB metadata
alone therefore cannot tell the two apart, and neither board can be found by
id only.

The discriminator used here is the chip itself: a soft reset followed by a read
of the CLRC663 version register answers ``0x1A`` on the NFC board and on
nothing else. Probing costs two bytes written and one read per candidate, and
is only ever done on ports matching the shared USB id.

This module is imported by ``daemon.utils.find_serial_port`` as well, to keep
the NFC board out of the motor controller's candidate list: without it, simply
plugging the reader into a Lite robot would make the daemon refuse to start
with "Multiple Reachy Mini serial ports found".
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Sequence

logger = logging.getLogger(__name__)

# CH343 USB-serial bridge. Shared with the motor controller — see module
# docstring; this is why every lookup here ends with a protocol probe.
NFC_VID = 0x1A86
NFC_PID = 0x55D3

# CLRC663 version register value. 0x18 is a CLRC66301/02, which this driver
# does not support.
CLRC663_VERSION = 0x1A

# Probe timeout. Short on purpose: the probe runs on the daemon's startup path
# when several candidate ports exist, and a missing answer is itself the answer.
PROBE_TIMEOUT = 0.2


def candidate_ports(
    comports: Sequence[Any] | None = None,
    exclude: Iterable[str] = (),
) -> list[str]:
    """Return serial ports whose USB ids match the NFC reader board.

    Args:
        comports: optional list of ports (for testing); defaults to the live
            ``serial.tools.list_ports.comports()``.
        exclude: ports to leave out (typically the motor controller's port,
            when it is known explicitly).

    """
    if comports is None:
        import serial.tools.list_ports

        comports = serial.tools.list_ports.comports()

    excluded = set(exclude)
    return [
        p.device
        for p in comports
        if p.vid == NFC_VID and p.pid == NFC_PID and p.device not in excluded
    ]


def probe_clrc663(port: str) -> bool | None:
    """Whether ``port`` is an NFC reader board.

    Returns ``True`` when the CLRC663 answers its version register, ``False``
    when the port opens but nothing answers (a motor controller, say), and
    ``None`` when the port cannot be opened at all — most often because the
    reader thread or the robot backend already holds it. The three cases must
    stay distinct: a port that cannot be opened is not a port that answered
    "not me".
    """
    try:
        from winnie_nfc.core import CLRC663
        from winnie_nfc.transport import Transport
    except ImportError:
        logger.debug("winnie_nfc not installed; cannot probe %s.", port)
        return None

    transport = Transport(port=port, timeout=PROBE_TIMEOUT)
    try:
        transport.open()
    except Exception as e:  # noqa: BLE001 - absent, busy or forbidden
        logger.debug("Cannot open %s to probe for an NFC reader: %s", port, e)
        return None

    try:
        chip = CLRC663(transport)
        chip.soft_reset()
        return bool(chip.version() == CLRC663_VERSION)
    except Exception as e:  # noqa: BLE001 - silence is a valid answer here
        logger.debug("No CLRC663 on %s: %s", port, e)
        return False
    finally:
        transport.close()


def find_nfc_port(
    comports: Sequence[Any] | None = None,
    exclude: Iterable[str] = (),
    probe: Callable[[str], bool | None] = probe_clrc663,
) -> str | None:
    """Return the port of the NFC reader board, or ``None`` if there is none."""
    for port in candidate_ports(comports=comports, exclude=exclude):
        if probe(port) is True:
            return port
    return None


def exclude_nfc_boards(
    ports: list[str],
    probe: Callable[[str], bool | None] = probe_clrc663,
) -> list[str]:
    """Drop NFC reader boards from a list of motor controller candidates.

    Only ports that positively answer "no CLRC663 here" are kept: a port that
    cannot even be opened is dropped too, since the motor controller needs to
    open it anyway, and a busy port is most likely the NFC reader already being
    served by its own thread.

    Falls back to the untouched list when that would leave nothing — better to
    let the caller report its own "no port found" than to silently claim the
    hardware is absent.
    """
    kept = [port for port in ports if probe(port) is False]
    if not kept:
        logger.debug("Probing left no motor candidate among %s.", ports)
        return ports
    if len(kept) != len(ports):
        logger.info(
            "Ignoring NFC reader board(s) %s when looking for the motor controller.",
            sorted(set(ports) - set(kept)),
        )
    return kept
