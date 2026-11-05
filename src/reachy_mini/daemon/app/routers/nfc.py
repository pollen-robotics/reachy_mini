"""NFC reader API routes.

``/tag`` and ``/status`` always answer. The routes that act on a tag answer
503 while the reader is off and report any other failure in their result.
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
)
from ... import nfc_switch
from ..dependencies import get_nfc_reader

router = APIRouter(prefix="/nfc")


def _refused(e: nfc_switch.NfcDisabledError) -> HTTPException:
    return HTTPException(status_code=503, detail=str(e))


def _reader(reader: NfcReader | None = Depends(get_nfc_reader)) -> NfcReader:
    """Get the NFC reader, or answer 503 when the daemon runs without one."""
    try:
        return nfc_switch.require_reader(reader)
    except nfc_switch.NfcDisabledError as e:
        raise _refused(e) from e


def _enabled_reader(reader: NfcReader | None = Depends(get_nfc_reader)) -> NfcReader:
    """Get the NFC reader, or answer 503 when it is off or was never created."""
    try:
        return nfc_switch.require_enabled(reader)
    except nfc_switch.NfcDisabledError as e:
        raise _refused(e) from e


@router.get("/tag")
async def get_tag(reader: NfcReader | None = Depends(get_nfc_reader)) -> NfcTag:
    """Get the tag currently on the reader.

    The ``uid`` is read on any ISO 14443-A tag, the NDEF ``records`` only on
    Type 2 tags. An unreadable tag has ``readable`` false and an ``error``.
    """
    return nfc_switch.tag(reader)


@router.get("/status")
async def get_status(
    reader: NfcReader | None = Depends(get_nfc_reader),
) -> NfcStatus:
    """Get the NFC reader hardware status (link, chip detection, driver)."""
    return nfc_switch.status(reader)


@router.post("/enabled")
async def set_enabled(
    request: NfcEnableRequest,
    reader: NfcReader = Depends(_reader),
) -> NfcStatus:
    """Switch the NFC reader on or off, and remember it across restarts.

    Off, the serial port is released. Answers with the status right after the
    switch; ``setting_saved`` is false when the choice could not be persisted.
    Answers 503 when the daemon was started with ``--no-nfc``.
    """
    return await asyncio.to_thread(nfc_switch.switch_reader, reader, request.enabled)


@router.get("/dump")
async def dump_tag(reader: NfcReader = Depends(_enabled_reader)) -> NfcDump:
    """Read the whole user memory of the tag on the reader, as hex.

    For tags carrying raw data rather than NDEF; ``GET /tag`` only reports its
    first bytes in ``raw_hex``.
    """
    return await asyncio.to_thread(reader.dump)


@router.post("/erase")
async def erase_tag(
    request: NfcEraseRequest,
    reader: NfcReader = Depends(_enabled_reader),
) -> NfcWriteResult:
    """Make the next tag presented to the reader blank again.

    By default, writes an empty NDEF message: the tag stays formatted. ``full``
    also zeroes the whole user memory (a few seconds). The capability
    container, lock bytes and password are never touched.
    """
    return await asyncio.to_thread(reader.erase, request)


@router.post("/write")
async def write_tag(
    request: NfcWriteRequest,
    reader: NfcReader = Depends(_enabled_reader),
) -> NfcWriteResult:
    """Write text or a URI onto the next tag presented to the reader.

    Exactly one of ``text`` or ``uri``. Waits for a tag to be presented. A
    message larger than the tag's capacity (144 bytes on an NTAG213, 496 on an
    NTAG215, 872 on an NTAG216) answers ``TOO_LONG``.
    """
    return await asyncio.to_thread(reader.write, request)
