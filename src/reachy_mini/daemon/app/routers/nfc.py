"""NFC reader API routes.

Exposes the optional Arduino + PN532 NFC reader as simple HTTP endpoints. These
routes never depend on the robot backend, so they work even when the robot is
not started — and they degrade gracefully when the reader is disabled or absent
(no error, just a "not connected / no tag" state).
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from ....nfc import (
    NfcReader,
    NfcStatus,
    NfcTag,
    NfcWriteRequest,
    NfcWriteResult,
)
from ..dependencies import get_nfc_reader

router = APIRouter(prefix="/nfc")


@router.get("/tag")
async def get_tag(reader: NfcReader | None = Depends(get_nfc_reader)) -> NfcTag:
    """Get the tag currently on the reader (presence, UID and written content)."""
    if reader is None:
        return NfcTag(
            present=False, uid=None, content=None, blank=False, last_read_at=None
        )
    return reader.get_tag()


@router.get("/status")
async def get_status(
    reader: NfcReader | None = Depends(get_nfc_reader),
) -> NfcStatus:
    """Get the NFC reader hardware status (connection, module detection)."""
    if reader is None:
        return NfcStatus(
            connected=False,
            module_detected=False,
            port=None,
            last_line=None,
            last_line_at=None,
            error="NFC reader disabled",
        )
    return reader.get_status()


@router.post("/write")
async def write_tag(
    request: NfcWriteRequest,
    reader: NfcReader | None = Depends(get_nfc_reader),
) -> NfcWriteResult:
    """Write text (1-12 ASCII chars) onto the next tag presented to the reader.

    Blocks until the firmware reports the outcome (or a timeout). Returns the
    result rather than raising, except when the reader is unavailable.
    """
    if reader is None:
        raise HTTPException(status_code=503, detail="NFC reader disabled")
    if not reader.is_connected():
        raise HTTPException(status_code=503, detail="NFC reader not connected")

    # write() blocks (waits for a tag to be presented), so run it off the event
    # loop to avoid stalling other requests.
    return await asyncio.get_event_loop().run_in_executor(
        None, reader.write, request.text
    )
