"""Example: read and write NFC tags through the daemon's REST API.

``NfcClient`` wraps the ``/api/nfc`` routes; run this file for a small demo.
Field meanings and error codes are documented on the daemon's models
(``reachy_mini.nfc``) and in its OpenAPI page.
"""

import time
from typing import Any, Optional

import requests


class NfcClient:
    """Minimal client around the daemon's ``/api/nfc`` routes."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        """Create a client. ``base_url`` is the daemon address."""
        self.base = f"{base_url.rstrip('/')}/api/nfc"

    def status(self) -> dict[str, Any]:
        """Return the reader status (connected, port, error, ...)."""
        return self._call("GET", "/status")

    def read_tag(self) -> dict[str, Any]:
        """Return the tag currently on the reader (``present`` false if none)."""
        return self._call("GET", "/tag")

    def wait_for_tag(self, timeout: float = 30.0) -> Optional[dict[str, Any]]:
        """Poll until a tag is present, then return it (None on timeout)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            tag = self.read_tag()
            if tag["present"]:
                return tag
            time.sleep(0.3)
        return None

    def write(
        self, text: Optional[str] = None, uri: Optional[str] = None
    ) -> dict[str, Any]:
        """Write text or a URI onto the next tag presented (within ~6 s).

        Returns ``{"success", "error", "detail"}``.
        """
        body = {"text": text} if uri is None else {"uri": uri}
        return self._call("POST", "/write", json=body, timeout=15)

    def erase(self, full: bool = False) -> dict[str, Any]:
        """Make the next tag presented blank again (``full`` wipes all memory)."""
        return self._call("POST", "/erase", json={"full": full}, timeout=25)

    def _call(
        self, method: str, path: str, timeout: float = 5, **kwargs: Any
    ) -> dict[str, Any]:
        r = requests.request(method, self.base + path, timeout=timeout, **kwargs)
        r.raise_for_status()  # 503 when the reader is switched off
        result: dict[str, Any] = r.json()
        return result


if __name__ == "__main__":
    nfc = NfcClient()
    status = nfc.status()
    if not status["connected"]:
        raise SystemExit(f"NFC reader not connected: {status['error']}")

    print("Present a tag...")
    tag = nfc.wait_for_tag(timeout=15)
    if tag is None:
        raise SystemExit("No tag detected.")
    print(f"UID {tag['uid']} ({tag['model']}): {tag['content'] or 'no text/URI'}")

    text = input("Text to write on the next tag (empty to skip): ").strip()
    if text:
        print("Present a tag...")
        result = nfc.write(text=text)
        print("Written." if result["success"] else f"Failed: {result['error']}")
