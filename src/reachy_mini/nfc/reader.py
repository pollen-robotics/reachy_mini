r"""Optional NFC reader accessory (Arduino Nano + PN532 over USB serial).

This service is fully decoupled from the robot backend: a background thread owns
the serial link, parses the firmware's line protocol, and keeps a thread-safe
snapshot of the latest tag state. The reader's absence or failure never blocks
the daemon — a missing or badly-plugged Arduino simply leaves the state as
"not connected", and the thread keeps retrying so the reader can be hot-plugged.

Firmware line protocol (see ``examples/arduino_nfc/README.md``), 115200 baud, 8N1,
each line terminated by ``\\n``:

Arduino -> daemon:
    READY                       module initialised (emitted once at boot)
    READ:<uid_hex>:<content>    tag present with content (content may contain ':')
    EMPTY:<uid_hex>             a blank tag is present (uid known, no content)
    NO_TAG                      tag removed (after ~2 s without a tag)
    WRITE_PENDING               (optional) WRITE command received, waiting for tag
    WRITE_OK                    write succeeded
    WRITE_FAIL:<reason>         write failed (NO_TAG / WRITE_ERROR / INVALID)
    NFC_ERROR:NOT_FOUND         PN532 module not detected (repeated while absent)

daemon -> Arduino:
    WRITE:<text>\\n              write <text> (1..12 chars) onto the next tag

The uid is uppercase hex without separators (e.g. ``04A1B2C3D4E5F6``) and never
contains ``:``; the daemon therefore splits the read payload on the first ``:``.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

import serial
import serial.tools.list_ports
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# The CH340 USB-serial chip used by the Arduino Nano shares its vendor id with
# the Reachy Mini motor controller, so we must exclude the robot's product id
# when auto-detecting the NFC reader's port.
DEFAULT_NFC_VID = 0x1A86
REACHY_MOTOR_PID = 0x55D3

MAX_WRITE_LEN = 12


class NfcTag(BaseModel):
    """Snapshot of the tag currently on the reader."""

    present: bool
    uid: str | None  # hardware UID (hex), None if no tag
    content: str | None  # text written on the tag, None if blank/no tag
    blank: bool  # a tag is present but has no content
    last_read_at: datetime | None


class NfcStatus(BaseModel):
    """Snapshot of the NFC reader hardware status."""

    connected: bool  # serial port is currently open
    module_detected: bool  # the PN532 module answered (not in NFC_ERROR state)
    port: str | None
    last_line: str | None  # last raw line received (handy for debugging)
    last_line_at: datetime | None
    error: str | None


class NfcWriteRequest(BaseModel):
    """Request body to write text onto the next presented tag."""

    text: str


class NfcWriteResult(BaseModel):
    """Result of a write attempt."""

    success: bool
    error: str | None = None


def find_nfc_ports(
    vid: int = DEFAULT_NFC_VID,
    exclude_pids: tuple[int, ...] = (REACHY_MOTOR_PID,),
    comports: list | None = None,
) -> list[str]:
    """Return candidate serial ports for the NFC reader.

    Matches USB devices by vendor id while excluding the Reachy Mini motor
    controller's product id (it shares the same vendor id).

    Args:
        vid: USB vendor id to match (default: CH340 ``0x1A86``).
        exclude_pids: product ids to ignore (default: the robot motor board).
        comports: optional list of ports (for testing); defaults to the live
            ``serial.tools.list_ports.comports()``.

    """
    ports = comports if comports is not None else serial.tools.list_ports.comports()
    return [p.device for p in ports if p.vid == vid and p.pid not in exclude_pids]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class NfcReader:
    """Background NFC reader service over a USB serial link."""

    def __init__(
        self,
        port: str = "auto",
        baudrate: int = 115200,
        retry_interval: float = 2.0,
        serial_timeout: float = 1.0,
    ) -> None:
        """Create the reader (does not open the serial link yet).

        Args:
            port: serial port path, or ``"auto"`` to auto-detect by USB id.
            baudrate: serial baudrate (firmware uses 115200).
            retry_interval: seconds to wait before retrying after a failure.
            serial_timeout: ``readline`` timeout, so the loop can check for stop.

        """
        self._port_setting = port
        self._baudrate = baudrate
        self._retry_interval = retry_interval
        self._serial_timeout = serial_timeout

        self._serial: serial.Serial | None = None
        self._serial_lock = threading.Lock()  # guards write / open / close
        self._thread: threading.Thread | None = None
        self._should_stop = threading.Event()

        # Tag / status state, guarded by _state_lock.
        self._state_lock = threading.Lock()
        self._present = False
        self._uid: str | None = None
        self._content: str | None = None
        self._blank = False
        self._last_read_at: datetime | None = None
        self._connected = False
        self._module_detected = False
        self._port: str | None = None
        self._last_line: str | None = None
        self._last_line_at: datetime | None = None
        self._error: str | None = None

        # Write coordination. Two events mirror the firmware handshake:
        # WRITE_PENDING (command accepted, waiting for a tag) then the final
        # WRITE_OK / WRITE_FAIL. The final result also sets _write_pending so a
        # write() that started waiting still proceeds (and stays compatible with
        # firmwares that skip the WRITE_PENDING ack).
        self._write_lock = threading.Lock()  # one write at a time
        self._write_pending = threading.Event()
        self._write_done = threading.Event()
        self._write_result: NfcWriteResult | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the background reader thread. Never blocks or raises fatally."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._should_stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="nfc-reader", daemon=True
        )
        self._thread.start()
        logger.info("NFC reader thread started (port=%s).", self._port_setting)

    def stop(self, timeout: float = 3.0) -> None:
        """Stop the reader thread and close the serial link."""
        self._should_stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("NFC reader thread did not stop in time.")
        self._close_serial()

    # -- public state accessors -------------------------------------------

    def get_tag(self) -> NfcTag:
        """Return a snapshot of the tag currently on the reader."""
        with self._state_lock:
            return NfcTag(
                present=self._present,
                uid=self._uid,
                content=self._content,
                blank=self._blank,
                last_read_at=self._last_read_at,
            )

    def get_status(self) -> NfcStatus:
        """Return a snapshot of the reader hardware status."""
        with self._state_lock:
            return NfcStatus(
                connected=self._connected,
                module_detected=self._module_detected,
                port=self._port,
                last_line=self._last_line,
                last_line_at=self._last_line_at,
                error=self._error,
            )

    def is_connected(self) -> bool:
        """Whether the serial link is currently open."""
        with self._state_lock:
            return self._connected

    # -- write -------------------------------------------------------------

    @staticmethod
    def validate_write_text(text: str) -> str | None:
        """Return an error reason if ``text`` is not writable, else ``None``."""
        if not 1 <= len(text) <= MAX_WRITE_LEN:
            return "INVALID"
        if not text.isascii():
            return "INVALID"
        return None

    def write(
        self, text: str, timeout: float = 6.0, ack_timeout: float = 2.5
    ) -> NfcWriteResult:
        """Write ``text`` onto the next tag presented to the reader.

        Follows the firmware handshake: send ``WRITE:<text>``, expect a
        ``WRITE_PENDING`` ack within ``ack_timeout`` (the firmware then waits up
        to ~5 s for a tag), then the final ``WRITE_OK`` / ``WRITE_FAIL`` within
        ``timeout``. The firmware blocks reads while waiting for a tag, so writes
        are serialised with a lock.

        Returns an error result (never raises) on validation failure, when the
        link is down (``NOT_CONNECTED``), when the board never acknowledges
        (``NO_ACK``) or when the outcome never arrives (``TIMEOUT``).
        """
        err = self.validate_write_text(text)
        if err is not None:
            return NfcWriteResult(success=False, error=err)

        with self._write_lock:
            self._write_result = None
            self._write_pending.clear()
            self._write_done.clear()
            if not self._send_line(f"WRITE:{text}"):
                return NfcWriteResult(success=False, error="NOT_CONNECTED")
            # Phase 1: the board must acknowledge (or already deliver a result).
            if not self._write_pending.wait(ack_timeout):
                return NfcWriteResult(success=False, error="NO_ACK")
            # Phase 2: wait for the actual write outcome.
            if not self._write_done.wait(timeout):
                return NfcWriteResult(success=False, error="TIMEOUT")
            return self._write_result or NfcWriteResult(
                success=False, error="UNKNOWN"
            )

    # -- background loop ---------------------------------------------------

    def _run(self) -> None:
        while not self._should_stop.is_set():
            try:
                self._open_serial()
            except Exception as e:  # noqa: BLE001 - device may be absent
                self._set_disconnected(str(e))
                logger.debug("NFC reader not available: %s", e)
                self._should_stop.wait(self._retry_interval)
                continue

            try:
                self._read_loop()
            except Exception as e:  # noqa: BLE001 - link may drop at any time
                logger.warning("NFC reader link lost: %s", e)
                self._set_disconnected(str(e))
                self._close_serial()
                self._should_stop.wait(self._retry_interval)

    def _open_serial(self) -> None:
        if self._port_setting == "auto":
            ports = find_nfc_ports()
            if not ports:
                raise RuntimeError("no NFC reader serial port found")
            if len(ports) > 1:
                logger.warning(
                    "Multiple NFC reader candidate ports %s; using %s.",
                    ports,
                    ports[0],
                )
            port = ports[0]
        else:
            port = self._port_setting

        ser = serial.Serial(port, self._baudrate, timeout=self._serial_timeout)
        with self._serial_lock:
            self._serial = ser
        with self._state_lock:
            self._connected = True
            self._port = port
            self._error = None
        logger.info("NFC reader connected on %s.", port)

    def _read_loop(self) -> None:
        assert self._serial is not None
        while not self._should_stop.is_set():
            raw = self._serial.readline()
            if not raw:
                continue  # readline timeout; loop to re-check should_stop
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                self._handle_line(line)

    def _send_line(self, line: str) -> bool:
        with self._serial_lock:
            ser = self._serial
            if ser is None or not ser.is_open:
                return False
            try:
                ser.write((line + "\n").encode("utf-8"))
                ser.flush()
                return True
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to send to NFC reader: %s", e)
                return False

    def _close_serial(self) -> None:
        with self._serial_lock:
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:  # noqa: BLE001
                    pass
                self._serial = None

    def _set_disconnected(self, error: str | None) -> None:
        with self._state_lock:
            self._connected = False
            self._module_detected = False
            self._present = False
            self._uid = None
            self._content = None
            self._blank = False
            self._error = error

    # -- line protocol parsing --------------------------------------------

    def _handle_line(self, line: str) -> None:
        """Update internal state from a single firmware line.

        Pure with respect to I/O (only mutates in-memory state), so it can be
        unit-tested directly.
        """
        now = _now()
        with self._state_lock:
            self._last_line = line
            self._last_line_at = now

            if line == "READY":
                self._module_detected = True
            elif line.startswith("READ:"):
                payload = line[len("READ:") :]
                uid, _, content = payload.partition(":")
                self._module_detected = True
                self._present = True
                self._blank = False
                self._uid = uid or None
                self._content = content
                self._last_read_at = now
            elif line.startswith("EMPTY"):
                # "EMPTY:<uid>" or legacy bare "EMPTY"
                uid = line[len("EMPTY:") :] if line.startswith("EMPTY:") else ""
                self._module_detected = True
                self._present = True
                self._blank = True
                self._uid = uid or None
                self._content = None
                self._last_read_at = now
            elif line == "NO_TAG":
                self._module_detected = True
                self._present = False
                self._blank = False
                self._uid = None
                self._content = None
            elif line == "NFC_ERROR:NOT_FOUND":
                self._module_detected = False
            elif line == "WRITE_PENDING":
                self._write_pending.set()  # command accepted, waiting for a tag
            elif line == "WRITE_OK":
                self._write_result = NfcWriteResult(success=True)
                self._write_pending.set()
                self._write_done.set()
            elif line.startswith("WRITE_FAIL:"):
                reason = line[len("WRITE_FAIL:") :]
                self._write_result = NfcWriteResult(success=False, error=reason)
                self._write_pending.set()
                self._write_done.set()
            else:
                logger.debug("Unknown NFC line: %r", line)
