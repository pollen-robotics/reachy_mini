"""NFC reader API routes.

Exposes the optional CLRC663 reader board as simple HTTP endpoints. These
routes never depend on the robot backend, so they work even when the robot is
not started. ``/tag`` and ``/status`` always answer, whatever the state of the
reader; the routes that talk to a tag answer 503 while the reader is switched
off, and report any other failure inside their result.
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from ....nfc import (
    NfcDump,
    NfcEnableRequest,
    NfcEraseRequest,
    NfcReader,
    NfcStatus,
    NfcTag,
    NfcWriteRequest,
    NfcWriteResult,
    driver_available,
    no_tag,
    set_nfc_enabled,
)
from ..dependencies import get_nfc_reader

router = APIRouter(prefix="/nfc")


def _enabled_reader(reader: NfcReader | None = Depends(get_nfc_reader)) -> NfcReader:
    """Get the NFC reader, or answer 503 when it is off or was never created."""
    if reader is None or not reader.is_enabled():
        raise HTTPException(status_code=503, detail="NFC reader disabled")
    return reader


@router.get("/tag")
async def get_tag(reader: NfcReader | None = Depends(get_nfc_reader)) -> NfcTag:
    """Get the tag currently on the reader.

    Always reports what is known: a tag whose content cannot be decoded still
    carries its ``uid``, with ``readable`` false and ``error`` saying why. The
    UID is available on any ISO 14443-A tag, the NDEF ``records`` only on the
    Type 2 families the driver supports.
    """
    if reader is None:
        return no_tag()
    return reader.get_tag()


@router.get("/status")
async def get_status(
    reader: NfcReader | None = Depends(get_nfc_reader),
) -> NfcStatus:
    """Get the NFC reader hardware status (link, chip detection, driver)."""
    if reader is None:
        return NfcStatus(
            connected=False,
            enabled=False,
            chip_detected=False,
            driver_available=driver_available(),
            error="NFC reader disabled",
        )
    return reader.get_status()


@router.post("/enabled")
async def set_enabled(
    request: NfcEnableRequest,
    reader: NfcReader | None = Depends(get_nfc_reader),
) -> NfcStatus:
    """Switch the NFC reader on or off, and remember it across restarts.

    Off, the reader thread stops and the serial port is released; on, it goes
    back to looking for the board. Answers with the status right after the
    switch: when switching on, the board is usually not connected yet.

    Not available when the daemon was started with ``--no-nfc``.
    """
    if reader is None:
        raise HTTPException(status_code=503, detail="NFC reader disabled")

    set_nfc_enabled(request.enabled)
    # stop() joins the reader thread, which can take up to a poll's worth of
    # serial I/O: keep it off the event loop.
    await asyncio.to_thread(reader.start if request.enabled else reader.stop)
    return reader.get_status()


@router.get("/dump")
async def dump_tag(reader: NfcReader = Depends(_enabled_reader)) -> NfcDump:
    """Read the whole user memory of the tag on the reader, as hex.

    For tags that carry a bare code rather than an NDEF message — a badge
    programmed with an ASCII identifier in page 4, say. ``GET /tag`` reports
    the first bytes of such a payload in ``raw_hex``; this returns all of it.

    Costs a full transfer (~130 ms on an NTAG215), hence a separate endpoint
    instead of a field served on every poll.
    """
    return await asyncio.to_thread(reader.dump)


@router.post("/erase")
async def erase_tag(
    request: NfcEraseRequest,
    reader: NfcReader = Depends(_enabled_reader),
) -> NfcWriteResult:
    """Make the next tag presented to the reader blank again.

    By default one page is written: an NDEF message of length zero. The tag
    stays formatted, so it is immediately rewritable and a phone still sees it
    as an (empty) NFC tag.

    ``full`` also zeroes the whole user memory — the only way to remove a
    payload that is not NDEF, such as a bare code written straight into
    page 4. It costs one write per page, so it takes a few seconds.

    Neither restores the capability container, the lock bytes or a password:
    those pages are one-time programmable. A formatted tag stays formatted.
    """
    return await asyncio.to_thread(reader.erase, request)


@router.post("/write")
async def write_tag(
    request: NfcWriteRequest,
    reader: NfcReader = Depends(_enabled_reader),
) -> NfcWriteResult:
    """Write text or a URI onto the next tag presented to the reader.

    Exactly one of ``text`` or ``uri``. Blocks until the reader reports the
    outcome (or a timeout), retrying meanwhile so the tag can be presented
    after the request is posted. Failures come back as an error code in the
    result; only a reader switched off answers 503.

    The tag's own declared capacity is authoritative: 144 bytes of NDEF on an
    NTAG213, 496 on an NTAG215, 872 on an NTAG216. A message that does not fit
    comes back as ``TOO_LONG`` without anything having been written.
    """
    # write() blocks until a tag is presented: keep it off the event loop.
    return await asyncio.to_thread(reader.write, request)
