"""Example: using the Reachy Mini NFC reader from Python via a small class.

The daemon exposes the NFC reader over its REST API, so reading/writing tags
from Python is just three HTTP calls — no serial, no threads, no SPI on your
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
    if nfc.write_tag("badge42"):              # present a tag within ~5 s
        print("écrit !")

Requires ``requests`` (already a Reachy Mini dependency).
Run the built-in demo with:  ``python nfc_client.py``
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class Tag:
    """Snapshot of what's currently on the reader."""

    present: bool
    uid: Optional[str]
    content: Optional[str]
    blank: bool
    last_read_at: Optional[str]


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
        return Tag(**r.json())

    def status(self) -> dict:
        """Return the reader status (connected, module_detected, last_line, ...)."""
        r = requests.get(f"{self.base}/api/nfc/status", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def write_tag(self, text: str) -> bool:
        """Write ``text`` (1-12 ASCII chars) onto the next tag; True on success.

        Blocks until the firmware reports the outcome (present a tag within
        ~5 s). Use :meth:`write_tag_result` if you need the failure reason.
        """
        return bool(self.write_tag_result(text).get("success"))

    def write_tag_result(self, text: str) -> dict:
        """Like :meth:`write_tag` but returns the full ``{success, error}`` dict."""
        # The daemon blocks up to ~8 s (tag wait + handshake), so allow more time.
        r = requests.post(
            f"{self.base}/api/nfc/write",
            json={"text": text},
            timeout=max(self.timeout, 12.0),
        )
        if r.status_code == 503:
            return {"success": False, "error": r.json().get("detail", "unavailable")}
        r.raise_for_status()
        return r.json()

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
    print(f"Lecteur : connecté={status['connected']} "
          f"module={status['module_detected']} "
          f"(dernière ligne: {status['last_line']!r})")
    if not status["connected"]:
        print("⚠️  Lecteur non connecté — branche l'Arduino et relance le daemon.")
        return

    print("Approche un tag…")
    tag = nfc.wait_for_tag(timeout=15)
    if tag is None:
        print("Aucun tag détecté.")
        return
    print(f"Tag : uid={tag.uid} contenu={tag.content!r} vierge={tag.blank}")

    text = input("Texte à écrire sur le prochain tag (vide = ne rien faire) : ").strip()
    if text:
        print("Présente un tag…")
        result = nfc.write_tag_result(text)
        print("✓ Écrit !" if result["success"] else f"✗ Échec : {result['error']}")


if __name__ == "__main__":
    _demo()
