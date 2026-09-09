"""NFC reader API routes.

Exposes the optional CLRC663 reader board as simple HTTP endpoints. These
routes never depend on the robot backend, so they work even when the robot is
not started — and they degrade gracefully when the reader is disabled, absent,
or its driver is not installed (no error, just a "not connected / no tag"
state).
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from ....nfc import (
    NfcDump,
    NfcEraseRequest,
    NfcReader,
    NfcStatus,
    NfcTag,
    NfcWriteRequest,
    NfcWriteResult,
    driver_available,
    no_tag,
)
from ..dependencies import get_nfc_reader

router = APIRouter(prefix="/nfc")


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
            chip_detected=False,
            driver_available=driver_available(),
            error="NFC reader disabled",
        )
    return reader.get_status()


@router.get("/dump")
async def dump_tag(
    reader: NfcReader | None = Depends(get_nfc_reader),
) -> NfcDump:
    """Read the whole user memory of the tag on the reader, as hex.

    For tags that carry a bare code rather than an NDEF message — a badge
    programmed with an ASCII identifier in page 4, say. ``GET /tag`` reports
    the first bytes of such a payload in ``raw_hex``; this returns all of it.

    Costs a full transfer (~130 ms on an NTAG215), hence a separate endpoint
    instead of a field served on every poll.
    """
    if reader is None:
        return NfcDump(present=False, error="NFC reader disabled")
    return await asyncio.get_event_loop().run_in_executor(None, reader.dump)


@router.post("/erase")
async def erase_tag(
    request: NfcEraseRequest,
    reader: NfcReader | None = Depends(get_nfc_reader),
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
    if reader is None:
        raise HTTPException(status_code=503, detail="NFC reader disabled")
    if not reader.is_connected():
        raise HTTPException(status_code=503, detail="NFC reader not connected")

    return await asyncio.get_event_loop().run_in_executor(None, reader.erase, request)


@router.post("/write")
async def write_tag(
    request: NfcWriteRequest,
    reader: NfcReader | None = Depends(get_nfc_reader),
) -> NfcWriteResult:
    """Write text or a URI onto the next tag presented to the reader.

    Exactly one of ``text`` or ``uri``. Blocks until the reader reports the
    outcome (or a timeout), retrying meanwhile so the tag can be presented
    after the request is posted. Returns the result rather than raising, except
    when the reader itself is unavailable.

    The tag's own declared capacity is authoritative: 144 bytes of NDEF on an
    NTAG213, 496 on an NTAG215, 872 on an NTAG216. A message that does not fit
    comes back as ``TOO_LONG`` without anything having been written.
    """
    if reader is None:
        raise HTTPException(status_code=503, detail="NFC reader disabled")
    if not reader.is_connected():
        raise HTTPException(status_code=503, detail="NFC reader not connected")

    # write() blocks (it waits for a tag to be presented), so run it off the
    # event loop to avoid stalling other requests.
    return await asyncio.get_event_loop().run_in_executor(None, reader.write, request)
