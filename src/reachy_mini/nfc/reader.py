"""Optional NFC reader accessory (CLRC663 board driven by ``winnie_nfc``).

The reader is fully decoupled from the robot backend: a background thread owns
the serial link and the chip, polls the antenna, and keeps a thread-safe
snapshot of the latest tag state. Its absence or failure never blocks the
daemon — an unplugged board simply leaves the state as "not connected", and the
thread keeps retrying so the reader can be hot-plugged.

Two levels of information come out of a tag, and they do not have the same
domain of validity:

* the **UID**, available on every ISO 14443-A tag — blank, locked, or of a
  family whose memory this driver cannot read. It is the robust primitive for
  mapping a tag to a robot behaviour.
* the **NDEF content**, available on Type 2 tags (NTAG213/215/216, Ultralight
  EV1). Richer, but conditional: a MIFARE Classic or a phone in card emulation
  answers a UID and nothing this driver can decode.

``NfcTag`` therefore always reports what is known rather than failing: a tag
whose content could not be read still carries its UID, with ``readable`` false
and ``error`` saying why.

The ``winnie_nfc`` driver is an optional dependency (``reachy_mini[nfc]``). It is
imported lazily, so a daemon without it starts normally and reports
``driver_available: false`` instead of failing.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Protocol

from pydantic import BaseModel, model_validator

from .ports import find_nfc_port

logger = logging.getLogger(__name__)

# Antenna polling period. A detection costs ~35 ms and a full NDEF read ~130,
# so 5 Hz is the practical ceiling of the hardware, not an arbitrary choice.
POLL_INTERVAL = 0.2

# Wait before retrying after a failure, so a missing board does not spin.
RETRY_INTERVAL = 2.0

# Consecutive polls without an answer before a tag is declared gone.
# Detection is intermittent at the RF level: measured on this hardware at 5 Hz,
# a tag sitting still on the antenna answers every poll in a good position but
# misses one to three polls out of twenty in a mediocre one. Publishing the
# first miss as "tag removed" would make presence flicker for a tag nobody
# touched, so absence has to be confirmed. Three polls is ~0.6 s at 5 Hz, in
# the same ballpark as the previous accessory's firmware (~2 s).
ABSENCE_POLLS = 3

# Attempts a dump gets before reporting no tag, for the same reason.
DUMP_ATTEMPTS = 3

# How long a write waits for a tag to be presented, mirroring the previous
# accessory's behaviour: the caller posts a write, then puts a tag on the
# antenna.
WRITE_TIMEOUT = 6.0

# An erase waits for a tag the same way. A full erase then writes every user
# page one at a time — over a hundred on an NTAG215, each paying the tag's own
# programming time — so it needs more room than a write.
ERASE_TIMEOUT = 15.0

# Sanity bound on a write request. The authoritative limit is the tag's own
# declared NDEF capacity (144 bytes on an NTAG213, 496 on an NTAG215, 872 on an
# NTAG216) and the driver enforces it; this only rejects the absurd before the
# request reaches the reader thread.
MAX_WRITE_CHARS = 860

# How much of a non-NDEF tag's memory to report in a tag snapshot. Tags are
# routinely programmed with a bare code — the very first one tried on this
# hardware carried the ASCII "FC7FB644" in page 4, with no NDEF structure at
# all — and calling such a tag blank would throw away the only thing it holds.
# The full memory is available through ``dump()`` and ``GET /api/nfc/dump``.
RAW_PREVIEW_BYTES = 32


class NfcRecord(BaseModel):
    """One NDEF record read from a tag.

    ``kind`` is ``"text"``, ``"uri"``, or ``"other"`` for a record this driver
    does not decode (a vCard, a MIME payload, an Android Application Record).
    Those are reported rather than dropped: a tag carrying one would otherwise
    be indistinguishable from a blank tag.
    """

    kind: str
    value: str | None = None
    type_name: str | None = None  # NDEF type name, for "other"
    data_hex: str | None = None  # raw payload as hex, for "other"


class NfcTag(BaseModel):
    """Snapshot of the tag currently on the reader."""

    present: bool
    uid: str | None = None  # hardware UID, uppercase hex, no separator
    model: str | None = None  # "NTAG215", "inconnu", ...
    records: list[NfcRecord] = []
    content: str | None = None  # first text/URI value, for simple consumers
    blank: bool = False  # a tag is present and carries no NDEF message
    readable: bool = False  # the content could be read and decoded
    writable: bool = False  # identified, formatted and not locked
    capacity: int | None = None  # bytes of NDEF the tag declares
    raw_hex: str | None = None  # first bytes of user memory, when not NDEF
    error: str | None = None  # why the content is unreadable, if it is
    last_read_at: datetime | None = None


class NfcStatus(BaseModel):
    """Snapshot of the NFC reader hardware status."""

    connected: bool  # the serial link is open and the chip answered
    chip_detected: bool  # the CLRC663 answered its version register
    driver_available: bool  # the winnie_nfc package is installed
    port: str | None = None
    chip_version: str | None = None
    error: str | None = None
    last_seen_at: datetime | None = None


class NfcWriteRequest(BaseModel):
    """Request body to write onto the next presented tag.

    Exactly one of ``text`` or ``uri``. A URI is what a phone acts on — tapping
    the tag offers to open it — while text is for robot-internal payloads.
    """

    text: str | None = None
    uri: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "NfcWriteRequest":
        given = [v for v in (self.text, self.uri) if v is not None]
        if len(given) != 1:
            raise ValueError("provide exactly one of 'text' or 'uri'")
        if not given[0]:
            raise ValueError("the value to write must not be empty")
        if len(given[0]) > MAX_WRITE_CHARS:
            raise ValueError(
                f"at most {MAX_WRITE_CHARS} characters (the tag's own capacity "
                f"applies too, and is smaller)"
            )
        return self


class NfcEraseRequest(BaseModel):
    """Request body to make the tag blank again.

    ``full`` also zeroes the whole user memory, which is what it takes to
    remove a payload that is not NDEF — a bare code written straight into
    page 4. Slower: one write per page.
    """

    full: bool = False


class NfcWriteResult(BaseModel):
    """Result of a write attempt.

    ``error`` is a stable code rather than a sentence: ``NO_TAG``,
    ``TOO_LONG``, ``LOCKED``, ``UNKNOWN_TAG``, ``WRITE_REFUSED``,
    ``WRITE_ERROR``, ``COLLISION``, ``NOT_CONNECTED``, ``LINK_LOST``,
    ``TIMEOUT``, ``DRIVER_MISSING``.
    """

    success: bool
    error: str | None = None
    detail: str | None = None  # the driver's own message, for logs and debug


class NfcDump(BaseModel):
    """The raw user memory of a tag, for tags that carry no NDEF message.

    Reading it costs a full transfer (~130 ms on an NTAG215), so it is served
    on demand rather than on every poll.
    """

    present: bool
    uid: str | None = None
    model: str | None = None
    size: int = 0
    hex: str | None = None
    error: str | None = None


def no_tag() -> NfcTag:
    """Return an empty tag snapshot, for when there is no reader at all."""
    return NfcTag(present=False)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def driver_available() -> bool:
    """Whether the optional ``winnie_nfc`` driver is installed."""
    try:
        import winnie_nfc  # noqa: F401
    except ImportError:
        return False
    return True


class NfcSession(Protocol):
    """A live link to the reader board, owned by a single thread."""

    chip_version: str | None

    def read_tag(self) -> NfcTag:
        """Poll the antenna once and return what is on it."""
        ...

    def write(self, text: str | None, uri: str | None) -> NfcWriteResult:
        """Write onto the tag currently on the antenna."""
        ...

    def erase(self, full: bool) -> NfcWriteResult:
        """Make the tag on the antenna blank again."""
        ...

    def dump(self) -> NfcDump:
        """Read the whole user memory of the tag on the antenna."""
        ...

    def close(self) -> None:
        """Switch the field off and release the serial port."""
        ...


class Clrc663Session:
    """An ``winnie_nfc`` reader kept open for the lifetime of the link.

    ``NTagReader`` is a context manager, but this session outlives any single
    statement: the thread opens it once and serves many polls and writes from
    it. Entering and exiting it explicitly is deliberate — it is what gets the
    field switched off and the port closed on the way out.
    """

    def __init__(self, port: str) -> None:
        """Open the serial link and initialise the chip."""
        from winnie_nfc.reader import NTagReader

        self._reader = NTagReader(port=port)
        self._reader.__enter__()
        self.chip_version = self._chip_version()
        # Last full read, kept so that a tag sitting on the antenna is not
        # re-read at every poll: a detection costs ~35 ms, a full NDEF read
        # ~130. Polling only detects, and reads content when the UID changes.
        self._last_uid: str | None = None
        self._last_tag: NfcTag | None = None

    def _chip_version(self) -> str | None:
        try:
            return f"0x{self._reader.chip_version():02X}"
        except Exception:  # noqa: BLE001 - a version is nice to have, not vital
            return None

    def close(self) -> None:
        """Switch the field off and close the port."""
        self._reader.__exit__(None, None, None)

    def read_tag(self) -> NfcTag:
        """Poll the antenna once.

        Only the detection happens on every poll; the content is read when the
        UID changes. That is what makes 5 Hz reachable — a detection costs
        ~35 ms against ~130 for a full memory transfer, and re-reading an
        unchanged tag five times a second would buy nothing.

        Transport failures propagate: they mean the link is gone, and the
        reader thread reconnects. Tag-level failures do not — they are
        reported inside the snapshot, because "a tag is there but I cannot
        read it" is information, not an outage.
        """
        from winnie_nfc.core import NoTagError, TagError

        try:
            uid = self._reader.uid().hex().upper()
        except NoTagError:
            self._forget()
            return NfcTag(present=False)
        except TagError as e:
            # A collision or a malformed frame: something is on the antenna,
            # but nothing usable came back.
            self._forget()
            return NfcTag(present=True, error=str(e), last_read_at=_now())

        if uid == self._last_uid and self._last_tag is not None:
            # Same tag as last time: refresh when it was last seen, keep what
            # was read from it.
            fresh = self._last_tag.model_copy(deep=True)
            fresh.last_read_at = _now()
            return fresh

        try:
            read = self._reader.read_tag()
        except NoTagError:
            # Removed between the detection and the read.
            self._forget()
            return NfcTag(present=False)
        except TagError as e:
            self._forget()
            return NfcTag(present=True, uid=uid, error=str(e), last_read_at=_now())

        tag = _snapshot(read)
        self._last_uid = tag.uid
        self._last_tag = tag
        return tag

    def _forget(self) -> None:
        self._last_uid = None
        self._last_tag = None

    def dump(self) -> NfcDump:
        """Read the whole user memory of the tag on the antenna."""
        from winnie_nfc.core import NoTagError, TagError

        try:
            read = self._reader.read_tag()
        except NoTagError:
            return NfcDump(present=False)
        except TagError as e:
            return NfcDump(present=True, error=str(e))
        return NfcDump(
            present=True,
            uid=read.uid.hex().upper(),
            model=getattr(read.info, "name", None),
            size=len(read.data),
            hex=read.data.hex(),
            error=read.error,
        )

    def write(self, text: str | None, uri: str | None) -> NfcWriteResult:
        """Write text or a URI onto the tag on the antenna."""
        if uri is not None:
            return self._tag_write(lambda r: r.write_uri(uri))
        assert text is not None  # guaranteed by NfcWriteRequest
        return self._tag_write(lambda r: r.write_text(text))

    def erase(self, full: bool) -> NfcWriteResult:
        """Make the tag on the antenna blank again.

        Its capability container is left alone — page 3 is one-time
        programmable, so a formatted tag stays formatted. That is what is
        wanted: the tag stays reusable, and a phone still recognises it as an
        (empty) NFC tag.
        """
        return self._tag_write(lambda r: r.erase(full=full))

    def _tag_write(self, operation: Callable[[Any], None]) -> NfcWriteResult:
        """Run one tag-modifying driver call, turning its failures into codes.

        Every write path shares this translation, so a failure cannot be
        reported one way by a write and another way by an erase.
        """
        from winnie_nfc import ndef, type2
        from winnie_nfc.core import CollisionError, NoTagError, TagError

        try:
            operation(self._reader)
        except NoTagError as e:
            return NfcWriteResult(success=False, error="NO_TAG", detail=str(e))
        except CollisionError as e:
            return NfcWriteResult(success=False, error="COLLISION", detail=str(e))
        except type2.MessageTooLongError as e:
            return NfcWriteResult(success=False, error="TOO_LONG", detail=str(e))
        except type2.TagLockedError as e:
            return NfcWriteResult(success=False, error="LOCKED", detail=str(e))
        except type2.UnknownModelError as e:
            return NfcWriteResult(success=False, error="UNKNOWN_TAG", detail=str(e))
        except type2.WriteRefusedError as e:
            return NfcWriteResult(success=False, error="WRITE_REFUSED", detail=str(e))
        except ndef.NdefError as e:
            # The driver refuses a message no supported tag could hold, before
            # touching the antenna.
            return NfcWriteResult(success=False, error="TOO_LONG", detail=str(e))
        except type2.Type2Error as e:
            return NfcWriteResult(success=False, error="WRITE_ERROR", detail=str(e))
        except TagError as e:
            # A tag that answers the SELECT but not a Type 2 READ — a MIFARE
            # Classic, say. This must be caught here: letting it escape would
            # look like a broken serial link to the reader thread, which would
            # then drop and reopen a perfectly healthy one.
            return NfcWriteResult(success=False, error="WRITE_ERROR", detail=str(e))
        return NfcWriteResult(success=True)


def _snapshot(read: Any) -> NfcTag:
    """Turn one ``NTagReader.read_tag`` result into an ``NfcTag``."""
    info = read.info
    converted = [_record(r) for r in read.records or []]
    content = next(
        (r.value for r in converted if r.kind in ("text", "uri") and r.value),
        None,
    )

    # Two ways of carrying no content, and they must not be confused. Either
    # there is no NDEF message at all — a factory tag, or one holding a bare
    # code written straight into page 4 — or there is a message with zero
    # records, which is what an erase leaves behind. Both read as blank, but
    # only the first can hide raw bytes worth reporting.
    raw: bytes = read.data
    no_message = read.records is None and read.error is None
    empty_message = read.records == []
    has_raw = no_message and any(raw)

    return NfcTag(
        present=True,
        uid=read.uid.hex().upper(),
        model=getattr(info, "name", None),
        capacity=getattr(info, "ndef_bytes", None),
        writable=bool(
            info is not None and info.ndef_bytes is not None and not info.read_only
        ),
        records=converted,
        content=content,
        blank=(no_message or empty_message) and not has_raw,
        readable=read.error is None,
        raw_hex=raw[:RAW_PREVIEW_BYTES].hex() if has_raw else None,
        error=read.error,
        last_read_at=_now(),
    )


# The driver's record kinds are "text", "uri" and "other" — the same names the
# HTTP API uses, so nothing is translated here. Anything unexpected is reported
# as "other" rather than dropped.
_KINDS = ("text", "uri", "other")


def _record(record: dict[str, Any]) -> NfcRecord:
    kind = record["type"] if record["type"] in _KINDS else "other"
    if kind == "other":
        payload: bytes = record.get("payload", b"")
        return NfcRecord(
            kind=kind,
            type_name=record.get("type_name"),
            data_hex=payload.hex(),
        )
    return NfcRecord(kind=kind, value=record["value"])


class _Job:
    """Work handed to the reader thread, and its outcome.

    Everything that talks to the chip goes through one of these: the chip has
    a single owner, and the CLRC663 register protocol has no way to interleave
    two conversations.
    """

    def __init__(self) -> None:
        self.done = threading.Event()

    def run(self, session: NfcSession) -> bool:
        """Do the work. Return False to be retried after a poll interval."""
        raise NotImplementedError

    def abandon(self, error: str) -> None:
        """Record an outcome for a caller nobody else will answer."""
        raise NotImplementedError


class _WriteJob(_Job):
    """A tag-modifying operation, retried while no tag has been presented yet.

    Holds the operation rather than its arguments so that a write and an erase
    share one retry policy: both are things a caller posts before putting a
    tag on the antenna.
    """

    def __init__(self, action: Callable[[NfcSession], NfcWriteResult], deadline: float):
        super().__init__()
        self.action = action
        self.deadline = deadline
        self.result: NfcWriteResult | None = None

    def run(self, session: NfcSession) -> bool:
        result = self.action(session)
        if result.success or result.error != "NO_TAG":
            self.result = result
            return True
        if time.monotonic() >= self.deadline:
            # Out of time, and the last thing that happened was "no tag": that
            # is the answer the caller needs to hear.
            self.result = result
            return True
        return False

    def abandon(self, error: str) -> None:
        if self.result is None:
            self.result = NfcWriteResult(success=False, error=error)


class _DumpJob(_Job):
    """A read of the whole user memory, served on demand."""

    def __init__(self, attempts: int = DUMP_ATTEMPTS) -> None:
        super().__init__()
        self.dump: NfcDump | None = None
        self._left = attempts

    def run(self, session: NfcSession) -> bool:
        dump = session.dump()
        self._left -= 1
        if dump.present or dump.error is not None or self._left <= 0:
            self.dump = dump
            return True
        # No tag answered, but detection misses a poll now and then: retry
        # rather than tell the caller the antenna is empty.
        return False

    def abandon(self, error: str) -> None:
        if self.dump is None:
            self.dump = NfcDump(present=False, error=error)


class NfcReader:
    """Background NFC reader service.

    The chip is owned by one thread from end to end: polls and writes are
    serialised through a queue rather than sharing the serial link, because the
    CLRC663 register protocol has no way to interleave two conversations.
    """

    def __init__(
        self,
        port: str = "auto",
        exclude_ports: Iterable[str] = (),
        poll_interval: float = POLL_INTERVAL,
        retry_interval: float = RETRY_INTERVAL,
        session_factory: Callable[[str], NfcSession] | None = None,
        port_resolver: Callable[[], str | None] | None = None,
        absence_polls: int = ABSENCE_POLLS,
    ) -> None:
        """Create the reader (does not open the serial link yet).

        Args:
            port: serial port path, or ``"auto"`` to probe for the board.
            exclude_ports: ports never to probe — the motor controller's, when
                it is known explicitly. Both boards share the same USB ids, so
                auto-detection probes candidates; see ``ports.py``.
            poll_interval: seconds between two antenna polls.
            retry_interval: seconds before retrying after a failure.
            session_factory: opens a session on a port (injectable for tests).
            port_resolver: returns the port to use (injectable for tests).
            absence_polls: consecutive polls without an answer before a tag is
                declared gone. Detection misses a poll now and then, so a
                single miss must not read as a removal.

        """
        self._port_setting = port
        self._exclude_ports = tuple(exclude_ports)
        self._poll_interval = poll_interval
        self._retry_interval = retry_interval
        self._absence_polls = max(1, absence_polls)
        self._session_factory = session_factory or Clrc663Session
        self._port_resolver = port_resolver or self._resolve_port

        self._thread: threading.Thread | None = None
        self._should_stop = threading.Event()
        self._jobs: queue.Queue[_Job] = queue.Queue()
        self._write_lock = threading.Lock()  # one chip conversation at a time

        self._lock = threading.Lock()  # guards the snapshot below
        self._tag = no_tag()
        self._connected = False
        self._chip_detected = False
        self._port: str | None = None
        self._chip_version: str | None = None
        self._error: str | None = None
        self._last_seen_at: datetime | None = None
        self._misses = 0  # consecutive polls with no tag

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the background reader thread. Never blocks or raises fatally."""
        if self._thread is not None and self._thread.is_alive():
            return
        if not driver_available():
            # Not an error: the driver is an optional extra. Say so once, in
            # the status, instead of retrying an import that cannot succeed.
            with self._lock:
                self._error = (
                    "winnie_nfc is not installed (pip install 'reachy_mini[nfc]')"
                )
            logger.info("NFC reader disabled: winnie_nfc is not installed.")
            return
        self._should_stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="nfc-reader", daemon=True
        )
        self._thread.start()
        logger.info("NFC reader thread started (port=%s).", self._port_setting)

    def stop(self, timeout: float = 3.0) -> None:
        """Stop the reader thread and release the serial link."""
        self._should_stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("NFC reader thread did not stop in time.")
            self._thread = None

    # -- public state accessors -------------------------------------------

    def get_tag(self) -> NfcTag:
        """Return a snapshot of the tag currently on the reader."""
        with self._lock:
            return self._tag.model_copy(deep=True)

    def get_status(self) -> NfcStatus:
        """Return a snapshot of the reader hardware status."""
        with self._lock:
            return NfcStatus(
                connected=self._connected,
                chip_detected=self._chip_detected,
                driver_available=driver_available(),
                port=self._port,
                chip_version=self._chip_version,
                error=self._error,
                last_seen_at=self._last_seen_at,
            )

    def is_connected(self) -> bool:
        """Whether the serial link is currently open."""
        with self._lock:
            return self._connected

    # -- write -------------------------------------------------------------

    def write(
        self, request: NfcWriteRequest, timeout: float = WRITE_TIMEOUT
    ) -> NfcWriteResult:
        """Write onto the next tag presented to the reader.

        Blocks until the outcome is known, retrying while no tag is on the
        antenna so the caller can post the write first and present the tag
        after. Returns an error result rather than raising.
        """
        if not driver_available():
            return NfcWriteResult(success=False, error="DRIVER_MISSING")
        if not self.is_connected():
            return NfcWriteResult(success=False, error="NOT_CONNECTED")

        return self._post_write(
            lambda session: session.write(request.text, request.uri), timeout
        )

    def erase(
        self, request: NfcEraseRequest, timeout: float = ERASE_TIMEOUT
    ) -> NfcWriteResult:
        """Make the next tag presented to the reader blank again.

        Same waiting behaviour as a write. A full erase is slow — one write per
        page, over a hundred of them on an NTAG215 — hence the longer default
        timeout.
        """
        if not driver_available():
            return NfcWriteResult(success=False, error="DRIVER_MISSING")
        if not self.is_connected():
            return NfcWriteResult(success=False, error="NOT_CONNECTED")

        return self._post_write(lambda session: session.erase(request.full), timeout)

    def _post_write(
        self,
        action: Callable[[NfcSession], NfcWriteResult],
        timeout: float,
    ) -> NfcWriteResult:
        """Hand one tag-modifying operation to the reader thread and wait."""
        with self._write_lock:
            job = _WriteJob(action=action, deadline=time.monotonic() + timeout)
            self._jobs.put(job)
            # The thread has until the deadline, plus one poll period to notice
            # the job and one more to answer it.
            if not job.done.wait(timeout + 2 * self._poll_interval + 1.0):
                return NfcWriteResult(success=False, error="TIMEOUT")
            return job.result or NfcWriteResult(success=False, error="LINK_LOST")

    def dump(self, timeout: float = 5.0) -> NfcDump:
        """Read the whole user memory of the tag currently on the antenna.

        Goes through the reader thread like a write does: the chip has a single
        owner, and a 130 ms transfer must not interleave with a poll.
        """
        if not driver_available():
            return NfcDump(present=False, error="DRIVER_MISSING")
        if not self.is_connected():
            return NfcDump(present=False, error="NOT_CONNECTED")

        with self._write_lock:
            job = _DumpJob()
            self._jobs.put(job)
            if not job.done.wait(timeout + self._poll_interval):
                return NfcDump(present=False, error="TIMEOUT")
            return job.dump or NfcDump(present=False, error="LINK_LOST")

    # -- background loop ---------------------------------------------------

    def _resolve_port(self) -> str | None:
        if self._port_setting != "auto":
            return self._port_setting
        return find_nfc_port(exclude=self._exclude_ports)

    def _run(self) -> None:
        while not self._should_stop.is_set():
            session: NfcSession | None = None
            try:
                port = self._port_resolver()
                if port is None:
                    raise RuntimeError("no NFC reader board found")
                session = self._session_factory(port)
                self._set_connected(port, session.chip_version)
                logger.info("NFC reader connected on %s.", port)
            except Exception as e:  # noqa: BLE001 - the board may be absent
                self._set_disconnected(str(e))
                logger.debug("NFC reader not available: %s", e)
                self._close(session)
                self._should_stop.wait(self._retry_interval)
                continue

            try:
                self._serve(session)
            except Exception as e:  # noqa: BLE001 - the link may drop anytime
                logger.warning("NFC reader link lost: %s", e)
                self._set_disconnected(str(e))
                self._should_stop.wait(self._retry_interval)
            finally:
                self._close(session)

        self._set_disconnected(None)

    def _serve(self, session: NfcSession) -> None:
        """Alternate between polling the antenna and serving queued jobs."""
        while not self._should_stop.is_set():
            try:
                job = self._jobs.get(timeout=self._poll_interval)
            except queue.Empty:
                self._poll(session)
                continue
            try:
                self._serve_job(session, job)
            finally:
                # Only bites if _serve_job raised: the link is going down, and
                # nobody else will answer this caller.
                job.abandon("LINK_LOST")
                job.done.set()

    def _serve_job(self, session: NfcSession, job: _Job) -> None:
        """Run one job, retrying it while it asks to be retried."""
        while not job.run(session):
            if self._should_stop.is_set():
                job.abandon("NOT_CONNECTED")
                return
            self._should_stop.wait(self._poll_interval)

    def _poll(self, session: NfcSession) -> None:
        tag = session.read_tag()
        with self._lock:
            self._chip_detected = True
            self._last_seen_at = _now()
            self._error = None

            if tag.present:
                self._misses = 0
                self._tag = tag
                return

            # No answer. Keep the last known tag until absence is confirmed,
            # so one missed detection does not read as a removal.
            self._misses += 1
            if self._misses >= self._absence_polls or not self._tag.present:
                self._tag = tag

    def _close(self, session: NfcSession | None) -> None:
        if session is None:
            return
        try:
            session.close()
        except Exception as e:  # noqa: BLE001 - closing must never raise
            logger.debug("Error while closing the NFC session: %s", e)

    def _set_connected(self, port: str, chip_version: str | None) -> None:
        with self._lock:
            self._connected = True
            self._chip_detected = True
            # The chip answered its version register to open the session, so
            # it has been seen just now — leaving this null until the first
            # poll would read as "never answered".
            self._last_seen_at = _now()
            self._port = port
            self._chip_version = chip_version
            self._error = None

    def _set_disconnected(self, error: str | None) -> None:
        with self._lock:
            self._connected = False
            self._misses = 0
            self._chip_detected = False
            self._tag = no_tag()
            self._error = error
