"""Optional NFC reader accessory (CLRC663 board driven by ``winnie_nfc``).

A background thread owns the serial link, polls the antenna and keeps a
thread-safe snapshot of the tag on it. It is independent of the robot backend
and keeps retrying while no board is plugged in.

The UID is read from any ISO 14443-A tag; the NDEF content only from Type 2
tags (NTAG213/215/216, Ultralight EV1). A tag whose content cannot be read
still reports its UID, with ``readable`` false and ``error`` saying why.

``winnie_nfc`` is an optional dependency (``reachy_mini[nfc]``), imported
lazily.
"""

from __future__ import annotations

import functools
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Concatenate,
    Generic,
    Iterable,
    Literal,
    ParamSpec,
    Protocol,
    TypeVar,
)

from pydantic import BaseModel, model_validator

from .ports import NfcBoard, describe_port, find_nfc_board

logger = logging.getLogger(__name__)

# Antenna polling period: a detection takes ~70 ms, a full read ~180 ms.
POLL_INTERVAL = 0.2

# Wait before retrying after a failure.
RETRY_INTERVAL = 2.0

# Consecutive missed polls before a tag is declared gone: detection is
# intermittent at the RF level.
ABSENCE_POLLS = 3

# Polls that re-read a tag whose content could not be read, before giving up
# until the tag is removed.
CONTENT_READ_POLLS = 5

# Attempts of a dump before reporting no tag.
DUMP_ATTEMPTS = 3

# How long a write waits for a tag to be presented.
WRITE_TIMEOUT = 6.0

# How long an erase waits for a tag; a full erase writes every user page.
ERASE_TIMEOUT = 15.0

# How long a job waits for the one ahead of it before answering BUSY.
JOB_LOCK_TIMEOUT = 1.0

# Upper bound on a write request; the tag's own capacity is checked by the
# driver.
MAX_WRITE_CHARS = 860

# Bytes of user memory reported in ``raw_hex`` for a tag without NDEF.
RAW_PREVIEW_BYTES = 32

# Why the reader is not serving (``NfcStatus.error_code``).
NfcStatusError = Literal[
    "DISABLED",  # switched off, or the daemon runs with --no-nfc
    "NO_BOARD",  # switched on, no board plugged in: the usual state
    "DRIVER_MISSING",  # the winnie_nfc package is not installed
    "PORT_CONFLICT",  # --nfc-port names the motor controller's port
    "LINK_ERROR",  # the board is there but the serial link failed
]

DISABLED_ERROR = "NFC reader disabled"
NO_BOARD_ERROR = "no NFC reader board found"
DRIVER_MISSING_ERROR = "winnie_nfc is not installed (pip install 'reachy_mini[nfc]')"


class NfcRecord(BaseModel):
    """One NDEF record read from a tag.

    ``kind`` is ``"text"``, ``"uri"``, or ``"other"`` for a record that is not
    decoded (vCard, MIME payload...), reported with its raw payload.
    """

    kind: str
    value: str | None = None
    type_name: str | None = None  # NDEF type name, for "other"
    data_hex: str | None = None  # raw payload as hex, for "other"


class NfcTag(BaseModel):
    """Snapshot of the tag currently on the reader."""

    present: bool
    uid: str | None = None  # hardware UID, uppercase hex, no separator
    model: str | None = None  # "NTAG215", ...
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
    enabled: bool  # the reader is switched on (see NfcReader.start/stop)
    chip_detected: bool  # the CLRC663 answered its version register
    driver_available: bool  # the winnie_nfc package is installed
    port: str | None = None
    board_version: str | None = None  # from the USB product string
    chip_version: str | None = None
    error: str | None = None  # why the reader is not serving, for people
    error_code: NfcStatusError | None = None  # the same, for clients
    last_seen_at: datetime | None = None
    # Answer to a switch on/off only: False when the choice could not be
    # persisted and will be lost at the next restart.
    setting_saved: bool | None = None


class NfcWriteRequest(BaseModel):
    """Request body to write onto the next presented tag: ``text`` or ``uri``."""

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


class NfcEnableRequest(BaseModel):
    """Switch the reader on or off."""

    enabled: bool


class NfcEraseRequest(BaseModel):
    """Request body to make the tag blank again (see ``POST /api/nfc/erase``)."""

    full: bool = False


class NfcWriteResult(BaseModel):
    """Result of a write attempt.

    ``error`` is a stable code rather than a sentence: ``NO_TAG``,
    ``TOO_LONG``, ``LOCKED``, ``UNKNOWN_TAG``, ``WRITE_REFUSED``,
    ``WRITE_ERROR``, ``COLLISION``, ``NOT_CONNECTED``, ``LINK_LOST``,
    ``TIMEOUT``, ``UNCERTAIN``, ``BUSY``, ``DRIVER_MISSING``.

    ``WRITE_REFUSED``: the tag refused a page (locked or password-protected
    page on a tag otherwise declared writable). When the refusal comes after
    the first page, the tag now reads as blank: its previous content is lost.
    The same applies to ``NO_TAG`` when the tag is removed during the write.
    ``TIMEOUT``: no tag came in time, nothing was written. ``UNCERTAIN``: the
    caller stopped waiting while the operation was running on the tag.
    """

    success: bool
    error: str | None = None
    detail: str | None = None  # the driver's own message, for logs and debug


class NfcDump(BaseModel):
    """The raw user memory of a tag (see ``GET /api/nfc/dump``)."""

    present: bool
    uid: str | None = None
    model: str | None = None
    size: int = 0
    hex: str | None = None
    error: str | None = None


def no_tag() -> NfcTag:
    """Return an empty tag snapshot, for when there is no reader at all."""
    return NfcTag(present=False)


def disabled_status() -> NfcStatus:
    """Return the status reported when there is no reader at all (--no-nfc)."""
    return NfcStatus(
        connected=False,
        enabled=False,
        chip_detected=False,
        driver_available=driver_available(),
        error=DISABLED_ERROR,
        error_code="DISABLED",
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


@functools.cache
def driver_available() -> bool:
    """Whether the optional ``winnie_nfc`` driver is importable (cached)."""
    try:
        import winnie_nfc  # noqa: F401
    except ImportError:
        return False
    except Exception as e:  # noqa: BLE001 - a broken install reads as missing
        logger.warning("winnie_nfc is installed but failed to import: %s", e)
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

    def forget(self) -> None:
        """Drop what was read from the last tag: its absence is confirmed."""
        ...

    def close(self) -> None:
        """Switch the field off and release the serial port."""
        ...


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _field_off_after(
    method: Callable[Concatenate[Clrc663Session, _P], _R],
) -> Callable[Concatenate[Clrc663Session, _P], _R]:
    """Switch the RF field off once the call is over, however it ends.

    A failure to switch it off is only logged, so it never replaces the call's
    own result.
    """

    @functools.wraps(method)
    def call(self: Clrc663Session, /, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return method(self, *args, **kwargs)
        finally:
            try:
                self._reader.field_off()
            except Exception as e:  # noqa: BLE001
                logger.debug("Could not switch the NFC field off: %r", e)

    return call


class Clrc663Session:
    """A ``winnie_nfc`` reader kept open for the lifetime of the link."""

    def __init__(self, port: str) -> None:
        """Open the serial link and initialise the chip."""
        from winnie_nfc.reader import NTagReader

        self._reader = NTagReader(port=port)
        self._reader.__enter__()
        self.chip_version = self._chip_version()
        # Last full read: a tag that stays on the antenna is not re-read.
        self._last_uid: str | None = None
        self._last_tag: NfcTag | None = None
        self._failed_reads = 0  # content reads that failed on _last_uid

    def _chip_version(self) -> str | None:
        try:
            return f"0x{self._reader.chip_version():02X}"
        except Exception:  # noqa: BLE001 - optional information
            return None

    def close(self) -> None:
        """Switch the field off and close the port."""
        self._reader.__exit__(None, None, None)

    @_field_off_after
    def read_tag(self) -> NfcTag:
        """Poll the antenna once.

        Every poll detects the UID; the content is read when the UID changes,
        and re-read up to ``CONTENT_READ_POLLS`` times while it fails. The
        cached read is kept until :meth:`forget`.

        Transport failures propagate (the reader thread reconnects); tag-level
        failures are reported in the snapshot.
        """
        from winnie_nfc.core import CollisionError, TagError

        try:
            uid = self._reader.uid().hex().upper()
        except CollisionError as e:
            # Several tags on the antenna.
            self.forget()
            return NfcTag(present=True, error=str(e), last_read_at=_now())
        except TagError:
            # No answer or a malformed frame: a missed detection.
            return no_tag()

        if uid != self._last_uid:
            self.forget()
        elif self._last_tag is not None and (
            self._last_tag.readable or self._failed_reads >= CONTENT_READ_POLLS
        ):
            # Same tag: keep its content, refresh its timestamp.
            fresh = self._last_tag.model_copy(deep=True)
            fresh.last_read_at = _now()
            return fresh

        try:
            read = self._reader.read_tag()
        except TagError:
            # Removed between the detection and the read.
            return no_tag()

        try:
            tag = _snapshot(read)
        except Exception as e:  # noqa: BLE001 - decoding error, not a link error
            logger.warning("Could not decode the NFC tag: %r", e)
            tag = NfcTag(
                present=True,
                uid=uid,
                readable=False,
                error=f"undecodable tag: {e!r}",
                last_read_at=_now(),
            )
        self._last_uid = tag.uid
        self._last_tag = tag
        self._failed_reads = 0 if tag.readable else self._failed_reads + 1
        return tag

    def forget(self) -> None:
        """Drop the cached read: the next detection reads the content again."""
        self._last_uid = None
        self._last_tag = None
        self._failed_reads = 0

    @_field_off_after
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
            uid=_uid(read),
            model=_model(read),
            size=len(read.data),
            hex=read.data.hex(),
            error=read.error,
        )

    @_field_off_after
    def write(self, text: str | None, uri: str | None) -> NfcWriteResult:
        """Write text or a URI onto the tag on the antenna."""
        if uri is not None:
            return self._tag_write(lambda r: r.write_uri(uri))
        assert text is not None  # guaranteed by NfcWriteRequest
        return self._tag_write(lambda r: r.write_text(text))

    @_field_off_after
    def erase(self, full: bool) -> NfcWriteResult:
        """Make the tag on the antenna blank again."""
        return self._tag_write(lambda r: r.erase(full=full))

    def _tag_write(self, operation: Callable[[Any], None]) -> NfcWriteResult:
        """Run one tag-modifying driver call, turning its failures into codes."""
        from winnie_nfc import ndef, type2
        from winnie_nfc.core import CollisionError, NoTagError, TagError
        from winnie_nfc.transport import TransportError

        # The tag keeps its UID: drop the cached read, whatever the outcome.
        self.forget()
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
            # Larger than any supported tag; refused before touching the tag.
            return NfcWriteResult(success=False, error="TOO_LONG", detail=str(e))
        except type2.Type2Error as e:
            return NfcWriteResult(success=False, error="WRITE_ERROR", detail=str(e))
        except TagError as e:
            # Not a Type 2 tag (MIFARE Classic...).
            return NfcWriteResult(success=False, error="WRITE_ERROR", detail=str(e))
        except (TransportError, OSError):
            # Link failure (SerialException is an OSError): the thread reconnects.
            raise
        except Exception as e:  # noqa: BLE001 - argument rejected by the driver
            logger.warning("Unexpected NFC write failure: %r", e)
            return NfcWriteResult(success=False, error="WRITE_ERROR", detail=repr(e))
        return NfcWriteResult(success=True)


def _uid(read: Any) -> str:
    """Return the UID of a ``read_tag`` result, as the API reports it."""
    uid: bytes = read.uid
    return uid.hex().upper()


def _model(read: Any) -> str | None:
    """Return the model of a ``read_tag`` result (None if unidentified)."""
    return read.info.name if read.info is not None else None


def _snapshot(read: Any) -> NfcTag:
    """Turn one ``NTagReader.read_tag`` result into an ``NfcTag``."""
    info = read.info
    converted = [_record(r) for r in read.records or []]
    content = next(
        (r.value for r in converted if r.kind in ("text", "uri") and r.value),
        None,
    )

    # No NDEF message (factory tag, or a raw code in page 4) or an empty one
    # (an erased tag): both are blank, but only the first can hold raw bytes.
    raw: bytes = read.data
    no_message = read.records is None and read.error is None
    empty_message = read.records == []
    has_raw = no_message and any(raw)

    return NfcTag(
        present=True,
        uid=_uid(read),
        model=_model(read),
        capacity=info.ndef_bytes if info is not None else None,
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


# Record kinds of the driver, kept as is; anything else is reported as "other".
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


_Outcome = TypeVar("_Outcome", NfcWriteResult, NfcDump)


class _Job(Generic[_Outcome]):
    """Work handed to the reader thread, and its outcome."""

    def __init__(self) -> None:
        self.done = threading.Event()
        self.outcome: _Outcome | None = None
        self._state = threading.Lock()
        self._running = False
        self._cancelled = False

    def begin(self) -> bool:
        """Claim the job for one attempt. False once the caller gave up."""
        with self._state:
            if self._cancelled:
                return False
            self._running = True
            return True

    def end(self) -> None:
        """Mark the current attempt as over."""
        with self._state:
            self._running = False

    def cancel(self) -> bool:
        """Withdraw the job unless an attempt is running. False if it is."""
        with self._state:
            if self._running:
                return False
            self._cancelled = True
            return True

    def wait(self, timeout: float) -> Literal["done", "withdrawn", "running"]:
        """Wait for the outcome; on timeout, make sure the job never runs again.

        Returns ``"done"``, ``"withdrawn"`` (timed out between two attempts) or
        ``"running"`` (an attempt is still running, outcome unknown).
        """
        if self.done.wait(timeout):
            return "done"
        if self.cancel():
            return "withdrawn"
        # An attempt is running: give it as long again.
        if self.done.wait(timeout):
            return "done"
        if self.cancel():
            return "withdrawn"
        logger.warning("NFC job still running after its caller gave up.")
        return "running"

    # Error reported when the caller gives up during an attempt.
    unsure_error = "TIMEOUT"

    def run(self, session: NfcSession) -> bool:
        """Do the work. Return False to be retried after a poll interval."""
        raise NotImplementedError

    def failed(self, error: str) -> _Outcome:
        """Build the outcome reporting ``error``, in this job's result type."""
        raise NotImplementedError

    def abandon(self, error: str) -> None:
        """Record an outcome for a caller nobody else will answer."""
        if self.outcome is None:
            self.outcome = self.failed(error)


class _WriteJob(_Job[NfcWriteResult]):
    """A write or an erase, retried while no tag has been presented yet."""

    unsure_error = "UNCERTAIN"

    def __init__(
        self, action: Callable[[NfcSession], NfcWriteResult], timeout: float
    ) -> None:
        super().__init__()
        self.action = action
        self.timeout = timeout
        # Set when the reader thread first runs the job.
        self.deadline: float | None = None

    def run(self, session: NfcSession) -> bool:
        if self.deadline is None:
            self.deadline = time.monotonic() + self.timeout
        result = self.action(session)
        # Only NO_TAG is retried, until the deadline.
        if (
            result.success
            or result.error != "NO_TAG"
            or time.monotonic() >= self.deadline
        ):
            self.outcome = result
            return True
        return False

    def failed(self, error: str) -> NfcWriteResult:
        return NfcWriteResult(success=False, error=error)


class _DumpJob(_Job[NfcDump]):
    """A read of the whole user memory, served on demand."""

    def __init__(self) -> None:
        super().__init__()
        self._left = DUMP_ATTEMPTS

    def run(self, session: NfcSession) -> bool:
        dump = session.dump()
        self._left -= 1
        if dump.present or dump.error is not None or self._left <= 0:
            self.outcome = dump
            return True
        return False

    def failed(self, error: str) -> NfcDump:
        return NfcDump(present=False, error=error)


class _BoardError(RuntimeError):
    """No usable board, for a reason the status reports as ``code``."""

    def __init__(self, code: NfcStatusError, message: str) -> None:
        super().__init__(message)
        self.code: NfcStatusError = code


class NfcReader:
    """Background NFC reader service.

    One thread owns the chip: polls and jobs (write, erase, dump) are
    serialised through a queue.
    """

    def __init__(
        self,
        port: str = "auto",
        exclude_ports: Iterable[str] = (),
        poll_interval: float = POLL_INTERVAL,
        retry_interval: float = RETRY_INTERVAL,
        session_factory: Callable[[str], NfcSession] | None = None,
        absence_polls: int = ABSENCE_POLLS,
    ) -> None:
        """Create the reader (does not open the serial link yet).

        Args:
            port: serial port path, or ``"auto"`` to look the board up by its
                USB product string.
            exclude_ports: ports never to take for the board (see ``ports.py``).
            poll_interval: seconds between two antenna polls.
            retry_interval: seconds before retrying after a failure.
            session_factory: opens a session on a port (injectable for tests).
            absence_polls: consecutive missed polls before a tag is declared
                gone.

        """
        self._port_setting = port
        self._exclude_ports = tuple(exclude_ports)
        self._poll_interval = poll_interval
        self._retry_interval = retry_interval
        self._absence_polls = max(1, absence_polls)
        self._session_factory = session_factory or Clrc663Session

        self._thread: threading.Thread | None = None
        self._should_stop = threading.Event()
        # Serialises start()/stop() with the thread's exit: one thread at most.
        self._lifecycle = threading.Lock()
        self._running = False  # a reader thread exists and has not exited
        self._enabled = False  # between start() and stop()
        self._jobs: queue.Queue[_Job[Any]] = queue.Queue()
        self._job_lock = threading.Lock()  # one job waited on at a time

        self._lock = threading.Lock()  # guards the snapshot below
        self._tag = no_tag()
        self._connected = False
        self._chip_detected = False
        self._port: str | None = None
        self._board_version: str | None = None
        self._chip_version: str | None = None
        self._error: str | None = None
        self._error_code: NfcStatusError | None = None
        self._last_seen_at: datetime | None = None
        self._misses = 0  # consecutive polls with no tag

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the background reader thread. Never blocks or raises fatally."""
        with self._lifecycle:
            self._enabled = True
            if self._running:
                # The thread has not exited yet: it keeps serving.
                self._should_stop.clear()
                return
            self._start_thread()

    def _start_thread(self) -> None:
        if not driver_available():
            with self._lock:
                self._error = DRIVER_MISSING_ERROR
                self._error_code = "DRIVER_MISSING"
            logger.info("NFC reader disabled: winnie_nfc is not installed.")
            return
        self._should_stop.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="nfc-reader", daemon=True
        )
        self._thread.start()
        logger.info("NFC reader thread started (port=%s).", self._port_setting)

    def stop(self, timeout: float = 3.0) -> None:
        """Stop the reader thread and release the serial link (restartable)."""
        with self._lifecycle:
            self._enabled = False
            self._should_stop.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                # It exits once its current job is done.
                logger.warning("NFC reader thread did not stop in time.")
        self._abandon_queued("NOT_CONNECTED")

    def _abandon_queued(self, error: str) -> None:
        """Answer every queued job with ``error``; none of them will run."""
        while True:
            try:
                job = self._jobs.get_nowait()
            except queue.Empty:
                return
            job.cancel()
            job.abandon(error)
            job.done.set()

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
                enabled=self._enabled,
                chip_detected=self._chip_detected,
                driver_available=driver_available(),
                port=self._port,
                board_version=self._board_version,
                chip_version=self._chip_version,
                error=self._error if self._enabled else DISABLED_ERROR,
                error_code=self._error_code if self._enabled else "DISABLED",
                last_seen_at=self._last_seen_at,
            )

    def is_enabled(self) -> bool:
        """Whether the reader is switched on (it may still have no board)."""
        return self._enabled

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
        antenna. Failures come back in the result.
        """
        return self._submit_write(lambda s: s.write(request.text, request.uri), timeout)

    def erase(
        self, request: NfcEraseRequest, timeout: float = ERASE_TIMEOUT
    ) -> NfcWriteResult:
        """Make the next tag presented to the reader blank again (see ``write``)."""
        return self._submit_write(lambda s: s.erase(request.full), timeout)

    def dump(self, timeout: float = 5.0) -> NfcDump:
        """Read the whole user memory of the tag currently on the antenna."""
        return self._submit(_DumpJob(), timeout + self._poll_interval)

    def _submit_write(
        self, action: Callable[[NfcSession], NfcWriteResult], timeout: float
    ) -> NfcWriteResult:
        """Submit a tag-modifying job that waits up to ``timeout`` for a tag."""
        wait = timeout + 2 * self._poll_interval + 1.0
        return self._submit(_WriteJob(action, timeout), wait)

    def _submit(self, job: _Job[_Outcome], wait: float) -> _Outcome:
        """Hand one job to the reader thread and wait up to ``wait`` for it.

        Failures come back as an outcome of the job's type, never raised.
        """
        if not driver_available():
            return job.failed("DRIVER_MISSING")
        if not self.is_connected():
            return job.failed("NOT_CONNECTED")
        if not self._job_lock.acquire(timeout=JOB_LOCK_TIMEOUT):
            return job.failed("BUSY")
        try:
            self._jobs.put(job)
            with self._lifecycle:
                if not self._running:
                    self._abandon_queued("NOT_CONNECTED")
            waited = job.wait(wait)
            if waited == "withdrawn":
                return job.failed("TIMEOUT")
            if waited == "running":
                return job.failed(job.unsure_error)
            return job.outcome if job.outcome is not None else job.failed("LINK_LOST")
        finally:
            self._job_lock.release()

    # -- background loop ---------------------------------------------------

    def _resolve_board(self) -> NfcBoard | None:
        if self._port_setting == "auto":
            return find_nfc_board(exclude=self._exclude_ports)
        if self._port_setting in self._exclude_ports:
            raise _BoardError(
                "PORT_CONFLICT",
                f"{self._port_setting} is the motor controller's port, "
                "not the NFC reader's",
            )
        return describe_port(self._port_setting)

    def _run(self) -> None:
        while True:
            self._run_until_stopped()
            with self._lifecycle:
                if self._should_stop.is_set():
                    self._running = False
                    self._set_disconnected(None, None)
                    self._abandon_queued("NOT_CONNECTED")
                    return
            # Restarted while finishing up: keep serving.

    def _run_until_stopped(self) -> None:
        while not self._should_stop.is_set():
            session: NfcSession | None = None
            try:
                board = self._resolve_board()
                if board is None:
                    raise _BoardError("NO_BOARD", NO_BOARD_ERROR)
                session = self._session_factory(board.port)
                self._set_connected(board, session.chip_version)
                logger.info(
                    "NFC reader connected on %s (board version %s).",
                    board.port,
                    board.version or "unknown",
                )
            except Exception as e:  # noqa: BLE001 - the board may be absent
                code = e.code if isinstance(e, _BoardError) else "LINK_ERROR"
                self._set_disconnected(str(e), code)
                self._abandon_queued("NOT_CONNECTED")
                logger.debug("NFC reader not available: %s", e)
                self._close(session)
                self._should_stop.wait(self._retry_interval)
                continue

            try:
                self._serve(session)
            except Exception as e:  # noqa: BLE001 - the link may drop anytime
                logger.warning("NFC reader link lost: %s", e)
                self._set_disconnected(str(e), "LINK_ERROR")
                self._abandon_queued("LINK_LOST")
                self._should_stop.wait(self._retry_interval)
            finally:
                self._close(session)

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
                # No-op unless _serve_job raised.
                job.abandon("LINK_LOST")
                job.done.set()

    def _serve_job(self, session: NfcSession, job: _Job[Any]) -> None:
        """Run one job, retrying it while it asks to be retried."""
        while True:
            if self._should_stop.is_set():
                job.abandon("NOT_CONNECTED")
                return
            if not job.begin():
                return  # the caller gave up
            try:
                finished = job.run(session)
            finally:
                job.end()
            if finished:
                return
            self._should_stop.wait(self._poll_interval)

    def _poll(self, session: NfcSession) -> None:
        tag = session.read_tag()
        if self._record_poll(tag):
            session.forget()

    def _record_poll(self, tag: NfcTag) -> bool:
        """Publish a poll's result. True when the tag's absence is confirmed."""
        with self._lock:
            self._chip_detected = True
            self._last_seen_at = _now()
            self._error = None
            self._error_code = None

            if tag.present:
                self._misses = 0
                self._tag = tag
                return False

            # Keep the last tag until its absence is confirmed.
            self._misses += 1
            if self._misses == self._absence_polls:
                self._tag = tag
                return True
            if not self._tag.present:
                self._tag = tag
            return False

    def _close(self, session: NfcSession | None) -> None:
        if session is None:
            return
        try:
            session.close()
        except Exception as e:  # noqa: BLE001 - closing must never raise
            logger.debug("Error while closing the NFC session: %s", e)

    def _set_connected(self, board: NfcBoard, chip_version: str | None) -> None:
        with self._lock:
            self._connected = True
            self._chip_detected = True
            self._last_seen_at = _now()
            self._port = board.port
            self._board_version = board.version
            self._chip_version = chip_version
            self._error = None
            self._error_code = None

    def _set_disconnected(
        self, error: str | None, error_code: NfcStatusError | None
    ) -> None:
        with self._lock:
            self._connected = False
            self._misses = 0
            self._chip_detected = False
            self._tag = no_tag()
            self._error = error
            self._error_code = error_code
