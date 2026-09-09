"""Example: using the Reachy Mini NFC reader from Python via a small class.

The daemon exposes the NFC reader over its REST API, so reading and writing
tags from Python is just a few HTTP calls — no serial, no threads on your
side. This module wraps those calls in a reusable ``NfcClient`` class you can
drop into any script::

    from nfc_client import NfcClient

    nfc = NfcClient()                         # default http://localhost:8000

    # --- read ---
    tag = nfc.read_tag()
    if tag.present:
        print("UID:", tag.uid, "contenu:", tag.content)

    # block until a tag shows up (or timeout)
    tag = nfc.wait_for_tag(timeout=10)

    # --- write ---
    if nfc.write_text("badge42"):             # present a tag within ~6 s
        print("écrit !")
    nfc.write_uri("https://pollen-robotics.com")   # a phone will open this

    # --- erase ---
    nfc.erase()                               # blank again, still formatted
    nfc.erase(full=True)                      # also wipes non-NDEF bytes

Two levels of information come back from a tag, with different reach:

* ``tag.uid`` is available on every ISO 14443-A tag, whatever its family and
  even when blank or locked. Map behaviours to the UID and they work on any
  tag you happen to own.
* ``tag.records`` (text and URI) needs a Type 2 tag — NTAG213/215/216 or
  Ultralight EV1. A tag whose content cannot be decoded still reports its UID,
  with ``readable=False`` and ``error`` saying why.

Requires ``requests`` (already a Reachy Mini dependency).
Run the built-in demo with:  ``python nfc_client.py``
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests


@dataclass
class Record:
    """One NDEF record. ``kind`` is ``"text"``, ``"uri"`` or ``"other"``."""

    kind: str
    value: Optional[str] = None
    type_name: Optional[str] = None
    data_hex: Optional[str] = None


@dataclass
class Tag:
    """Snapshot of what is currently on the reader."""

    present: bool
    uid: Optional[str] = None
    model: Optional[str] = None
    records: list[Record] = field(default_factory=list)
    content: Optional[str] = None
    blank: bool = False
    readable: bool = False
    writable: bool = False
    capacity: Optional[int] = None
    error: Optional[str] = None
    last_read_at: Optional[str] = None

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "Tag":
        """Build a tag from the daemon's JSON response."""
        records = [Record(**r) for r in payload.get("records", [])]
        return cls(**{**payload, "records": records})


class NfcClient:
    """Minimal client around the daemon's ``/api/nfc`` routes."""

    def __init__(
        self, base_url: str = "http://localhost:8000", timeout: float = 10.0
    ) -> None:
        """Create a client. ``base_url`` is the daemon address."""
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def read_tag(self) -> Tag:
        """Return the tag currently on the reader (``present=False`` if none)."""
        r = requests.get(f"{self.base}/api/nfc/tag", timeout=self.timeout)
        r.raise_for_status()
        return Tag.from_json(r.json())

    def status(self) -> dict[str, Any]:
        """Return the reader status (connected, chip_detected, port, ...)."""
        r = requests.get(f"{self.base}/api/nfc/status", timeout=self.timeout)
        r.raise_for_status()
        result: dict[str, Any] = r.json()
        return result

    def write_text(self, text: str) -> bool:
        """Write ``text`` onto the next tag presented; True on success.

        Blocks until the reader reports the outcome (present a tag within
        ~6 s). Use :meth:`write_result` if you need the failure reason.
        """
        return bool(self.write_result(text=text).get("success"))

    def write_uri(self, uri: str) -> bool:
        """Write ``uri`` onto the next tag presented; True on success.

        This is the record a phone acts on: tapping the tag offers to open the
        URL.
        """
        return bool(self.write_result(uri=uri).get("success"))

    def write_result(
        self, text: Optional[str] = None, uri: Optional[str] = None
    ) -> dict[str, Any]:
        """Write text or a URI and return the full ``{success, error}`` dict.

        Error codes are stable: ``NO_TAG``, ``TOO_LONG``, ``LOCKED``,
        ``UNKNOWN_TAG``, ``WRITE_REFUSED``, ``WRITE_ERROR``, ``COLLISION``,
        ``NOT_CONNECTED``, ``LINK_LOST``, ``TIMEOUT``, ``DRIVER_MISSING``.
        """
        body = {"text": text} if uri is None else {"uri": uri}
        # The daemon waits for a tag (~6 s) before answering, so allow more
        # time than a plain request.
        r = requests.post(
            f"{self.base}/api/nfc/write",
            json=body,
            timeout=max(self.timeout, 12.0),
        )
        if r.status_code == 503:
            return {
                "success": False,
                "error": r.json().get("detail", "unavailable"),
            }
        r.raise_for_status()
        result: dict[str, Any] = r.json()
        return result

    def erase(self, full: bool = False) -> bool:
        """Make the next tag presented blank again; True on success.

        One page is written by default: an NDEF message of length zero. The
        tag stays formatted, so it is immediately rewritable and a phone still
        sees it as an (empty) NFC tag.

        ``full=True`` also zeroes the whole user memory, which is the only way
        to remove a payload that is not NDEF — a bare code written straight
        into page 4. It costs one write per page, so allow a few seconds.

        Neither restores the capability container, the lock bytes or a
        password: those pages are one-time programmable.
        """
        return bool(self.erase_result(full=full).get("success"))

    def erase_result(self, full: bool = False) -> dict[str, Any]:
        """Like :meth:`erase` but returns the full ``{success, error}`` dict."""
        r = requests.post(
            f"{self.base}/api/nfc/erase",
            json={"full": full},
            timeout=max(self.timeout, 25.0),
        )
        if r.status_code == 503:
            return {
                "success": False,
                "error": r.json().get("detail", "unavailable"),
            }
        r.raise_for_status()
        result: dict[str, Any] = r.json()
        return result

    def wait_for_tag(self, timeout: float = 30.0, poll: float = 0.3) -> Optional[Tag]:
        """Poll until a tag is present, then return it (or None on timeout)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            tag = self.read_tag()
            if tag.present:
                return tag
            time.sleep(poll)
        return None


def _demo() -> None:
    """Small interactive demo: wait for a tag, then offer to write one."""
    nfc = NfcClient()

    status = nfc.status()
    print(
        f"Lecteur : connecté={status['connected']} "
        f"puce={status['chip_version']} port={status['port']}"
    )
    if not status["driver_available"]:
        print("⚠️  Driver absent — installe reachy_mini[nfc] et relance le daemon.")
        return
    if not status["connected"]:
        print(f"⚠️  Lecteur non connecté ({status['error']}).")
        return

    print("Approche un tag…")
    tag = nfc.wait_for_tag(timeout=15)
    if tag is None:
        print("Aucun tag détecté.")
        return

    print(f"Tag : uid={tag.uid} modèle={tag.model} capacité={tag.capacity}")
    if tag.blank:
        print("  tag vierge")
    elif not tag.readable:
        print(f"  contenu illisible : {tag.error}")
    for record in tag.records:
        detail = record.value or f"{record.type_name} {record.data_hex}"
        print(f"  {record.kind:6} : {detail}")

    text = input("Texte à écrire sur le prochain tag (vide = ne rien faire) : ").strip()
    if text:
        print("Présente un tag…")
        result = nfc.write_result(text=text)
        print("✓ Écrit !" if result["success"] else f"✗ Échec : {result['error']}")


if __name__ == "__main__":
    _demo()
