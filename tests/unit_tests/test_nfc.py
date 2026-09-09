"""Unit tests for the optional NFC reader accessory.

No hardware and no ``winnie_nfc`` install needed: the serial-port probe and the
reader session are both injected. What is under test is the part that has to
be right whatever the tag — port disambiguation against the motor controller,
and the reader thread's state machine.
"""

import threading
import time
from types import SimpleNamespace

import pytest

from reachy_mini.nfc import ports
from reachy_mini.nfc.reader import (
    NfcDump,
    NfcEraseRequest,
    NfcReader,
    NfcTag,
    NfcWriteRequest,
    NfcWriteResult,
    _snapshot,
)


def comport(device, vid=ports.NFC_VID, pid=ports.NFC_PID):
    """A fake entry of serial.tools.list_ports.comports()."""
    return SimpleNamespace(device=device, vid=vid, pid=pid)


# --------------------------------------------------------------- port lookup


def test_candidate_ports_matches_the_shared_usb_ids():
    found = ports.candidate_ports(
        comports=[
            comport("/dev/ttyACM0"),
            comport("/dev/ttyUSB0", vid=0x2886, pid=0x001A),  # ReSpeaker
            comport("/dev/ttyACM1"),
        ]
    )
    assert found == ["/dev/ttyACM0", "/dev/ttyACM1"]


def test_candidate_ports_honours_exclusions():
    found = ports.candidate_ports(
        comports=[comport("/dev/ttyACM0"), comport("/dev/ttyACM1")],
        exclude=["/dev/ttyACM0"],
    )
    assert found == ["/dev/ttyACM1"]


def test_find_nfc_port_returns_the_port_that_answers():
    # Both boards share their USB ids, so only the chip's own answer tells
    # them apart: the motor controller stays silent.
    answers = {"/dev/ttyACM0": False, "/dev/ttyACM1": True}
    found = ports.find_nfc_port(
        comports=[comport("/dev/ttyACM0"), comport("/dev/ttyACM1")],
        probe=answers.get,
    )
    assert found == "/dev/ttyACM1"


def test_find_nfc_port_returns_none_when_nothing_answers():
    assert (
        ports.find_nfc_port(
            comports=[comport("/dev/ttyACM0")], probe=lambda port: False
        )
        is None
    )


def test_exclude_nfc_boards_keeps_the_motor_controller():
    answers = {"/dev/ttyACM0": True, "/dev/ttyACM1": False}
    kept = ports.exclude_nfc_boards(["/dev/ttyACM0", "/dev/ttyACM1"], probe=answers.get)
    assert kept == ["/dev/ttyACM1"]


def test_exclude_nfc_boards_drops_a_port_it_cannot_open():
    # None means "could not even open": most likely the reader thread already
    # holds it. Keeping it would offer the accessory to the motor controller.
    answers = {"/dev/ttyACM0": None, "/dev/ttyACM1": False}
    kept = ports.exclude_nfc_boards(["/dev/ttyACM0", "/dev/ttyACM1"], probe=answers.get)
    assert kept == ["/dev/ttyACM1"]


def test_exclude_nfc_boards_falls_back_rather_than_claiming_nothing():
    # Probing left no candidate: better to hand the list back untouched and
    # let the caller report its own error than to pretend the robot is absent.
    kept = ports.exclude_nfc_boards(["/dev/ttyACM0"], probe=lambda port: True)
    assert kept == ["/dev/ttyACM0"]


# ------------------------------------------------------------- fake session


class FakeSession:
    """A reader session whose answers the test dictates.

    ``tags`` overrides ``tag`` and is consumed one snapshot per poll, the last
    one repeating — enough to script an intermittent detection.
    """

    def __init__(
        self,
        tag=None,
        write_results=None,
        fail_on_poll=None,
        tags=None,
        dumps=None,
        erases=None,
    ):
        self.chip_version = "0x1A"
        self.tag = tag if tag is not None else NfcTag(present=False)
        self.write_calls = []
        self.write_results = list(write_results or [])
        self.tags = list(tags or [])
        self.dump_results = list(dumps or [])
        self.erases = []
        self.erase_results = list(erases or [])
        self.fail_on_poll = fail_on_poll
        self.polls = 0
        self.dumps = 0
        self.closed = False

    def read_tag(self):
        self.polls += 1
        if self.fail_on_poll is not None and self.polls >= self.fail_on_poll:
            raise OSError("link dropped")
        if self.tags:
            return self.tags.pop(0) if len(self.tags) > 1 else self.tags[0]
        return self.tag

    def write(self, text, uri):
        self.write_calls.append((text, uri))
        if self.write_results:
            return self.write_results.pop(0)
        return NfcWriteResult(success=True)

    def erase(self, full):
        self.erases.append(full)
        if self.erase_results:
            return (
                self.erase_results.pop(0)
                if len(self.erase_results) > 1
                else self.erase_results[0]
            )
        return NfcWriteResult(success=True)

    def dump(self):
        self.dumps += 1
        if self.dump_results:
            return (
                self.dump_results.pop(0)
                if len(self.dump_results) > 1
                else self.dump_results[0]
            )
        return NfcDump(present=True, uid="04", size=4, hex="deadbeef")

    def close(self):
        self.closed = True


@pytest.fixture
def driver_installed(monkeypatch):
    """Pretend the optional winnie_nfc driver is installed."""
    monkeypatch.setattr("reachy_mini.nfc.reader.driver_available", lambda: True)


def started(reader):
    """Start a reader and wait until it reports being connected."""
    reader.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if reader.is_connected():
            return
        time.sleep(0.01)
    raise AssertionError("the reader never connected")


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def make_reader(session, port="/dev/ttyACM0", **kwargs):
    return NfcReader(
        port=port,
        poll_interval=0.01,
        retry_interval=0.01,
        session_factory=lambda p: session,
        port_resolver=lambda: port,
        **kwargs,
    )


# --------------------------------------------------------------- the service


def test_reader_publishes_the_polled_tag(driver_installed):
    tag = NfcTag(
        present=True, uid="04A1B2", model="NTAG215", readable=True, content="salut"
    )
    reader = make_reader(FakeSession(tag=tag))
    started(reader)
    try:
        assert wait_until(lambda: reader.get_tag().present)
        snapshot = reader.get_tag()
        assert (snapshot.uid, snapshot.content) == ("04A1B2", "salut")
        status = reader.get_status()
        assert status.connected and status.chip_detected
        assert status.chip_version == "0x1A"
        assert status.port == "/dev/ttyACM0"
    finally:
        reader.stop()


def test_snapshot_is_a_copy_and_cannot_be_mutated_from_outside(
    driver_installed,
):
    reader = make_reader(FakeSession(tag=NfcTag(present=True, uid="04")))
    started(reader)
    try:
        assert wait_until(lambda: reader.get_tag().present)
        reader.get_tag().uid = "tampered"
        assert reader.get_tag().uid == "04"
    finally:
        reader.stop()


def test_stopping_clears_the_state(driver_installed):
    session = FakeSession(tag=NfcTag(present=True, uid="04"))
    reader = make_reader(session)
    started(reader)
    assert wait_until(lambda: reader.get_tag().present)
    reader.stop()
    assert not reader.is_connected()
    assert not reader.get_tag().present
    assert session.closed


def test_write_is_served_by_the_reader_thread(driver_installed):
    session = FakeSession()
    reader = make_reader(session)
    started(reader)
    try:
        result = reader.write(NfcWriteRequest(text="badge42"))
        assert result.success
        assert session.write_calls == [("badge42", None)]
    finally:
        reader.stop()


def test_write_retries_while_no_tag_is_presented(driver_installed):
    # The caller posts the write, then puts a tag on the antenna: the first
    # attempts legitimately fail with NO_TAG and must not end the write.
    session = FakeSession(
        write_results=[
            NfcWriteResult(success=False, error="NO_TAG"),
            NfcWriteResult(success=False, error="NO_TAG"),
            NfcWriteResult(success=True),
        ]
    )
    reader = make_reader(session)
    started(reader)
    try:
        assert reader.write(NfcWriteRequest(uri="https://pollen.com")).success
        assert len(session.write_calls) == 3
        assert session.write_calls[-1] == (None, "https://pollen.com")
    finally:
        reader.stop()


def test_write_gives_up_at_its_deadline(driver_installed):
    session = FakeSession(
        write_results=[NfcWriteResult(success=False, error="NO_TAG")] * 50
    )
    reader = make_reader(session)
    started(reader)
    try:
        result = reader.write(NfcWriteRequest(text="x"), timeout=0.05)
        assert not result.success
        assert result.error == "NO_TAG"
    finally:
        reader.stop()


def test_a_definitive_refusal_is_not_retried(driver_installed):
    # A tag too small, or locked, will not become writable by waiting.
    session = FakeSession(
        write_results=[NfcWriteResult(success=False, error="TOO_LONG")]
    )
    reader = make_reader(session)
    started(reader)
    try:
        result = reader.write(NfcWriteRequest(text="x" * 500))
        assert result.error == "TOO_LONG"
        assert len(session.write_calls) == 1
    finally:
        reader.stop()


def test_write_without_a_link_answers_instead_of_blocking(driver_installed):
    reader = NfcReader(
        port="/dev/ttyACM0",
        poll_interval=0.01,
        retry_interval=0.01,
        session_factory=lambda p: FakeSession(),
        port_resolver=lambda: None,  # no board found
    )
    reader.start()
    try:
        assert not wait_until(reader.is_connected, timeout=0.2)
        result = reader.write(NfcWriteRequest(text="x"))
        assert result == NfcWriteResult(success=False, error="NOT_CONNECTED")
    finally:
        reader.stop()


def test_the_reader_reconnects_after_a_link_loss(driver_installed):
    sessions = []

    def factory(port):
        # The first session dies on its second poll, the next one holds.
        session = FakeSession(
            tag=NfcTag(present=True, uid="04"),
            fail_on_poll=2 if not sessions else None,
        )
        sessions.append(session)
        return session

    reader = NfcReader(
        port="/dev/ttyACM0",
        poll_interval=0.01,
        retry_interval=0.01,
        session_factory=factory,
        port_resolver=lambda: "/dev/ttyACM0",
    )
    reader.start()
    try:
        assert wait_until(lambda: len(sessions) >= 2)
        assert sessions[0].closed, "the dead session must be released"
        assert wait_until(lambda: reader.is_connected())
    finally:
        reader.stop()


def test_a_missing_driver_disables_the_reader_without_failing(monkeypatch):
    monkeypatch.setattr("reachy_mini.nfc.reader.driver_available", lambda: False)
    reader = make_reader(FakeSession())
    reader.start()  # must not raise, must not start a thread
    try:
        assert not reader.is_connected()
        status = reader.get_status()
        assert not status.driver_available
        assert status.error is not None and "winnie_nfc" in status.error
        assert reader.write(NfcWriteRequest(text="x")).error == "DRIVER_MISSING"
        assert threading.active_count() >= 1
    finally:
        reader.stop()


def test_a_single_missed_detection_is_not_a_removal(driver_installed):
    # Detection is intermittent at the RF level: a tag nobody touched misses a
    # poll now and then, and publishing that as a removal would make presence
    # flicker.
    here = NfcTag(present=True, uid="04", readable=True)
    gone = NfcTag(present=False)
    session = FakeSession(tags=[here, gone, here, here, here])
    reader = make_reader(session, absence_polls=3)
    started(reader)
    try:
        assert wait_until(lambda: session.polls >= 5)
        assert reader.get_tag().present, "one miss must not clear the tag"
        assert reader.get_tag().uid == "04"
    finally:
        reader.stop()


def test_a_confirmed_absence_clears_the_tag(driver_installed):
    here = NfcTag(present=True, uid="04", readable=True)
    gone = NfcTag(present=False)
    session = FakeSession(tags=[here, gone])
    reader = make_reader(session, absence_polls=3)
    started(reader)
    try:
        assert wait_until(lambda: session.polls >= 1 and reader.get_tag().present)
        assert wait_until(lambda: not reader.get_tag().present)
    finally:
        reader.stop()


def test_a_dump_retries_a_missed_detection(driver_installed):
    session = FakeSession(
        dumps=[
            NfcDump(present=False),
            NfcDump(present=True, uid="04", size=4, hex="deadbeef"),
        ]
    )
    reader = make_reader(session)
    started(reader)
    try:
        dump = reader.dump()
        assert dump.present and dump.hex == "deadbeef"
        assert session.dumps == 2
    finally:
        reader.stop()


def test_a_dump_gives_up_after_its_attempts(driver_installed):
    session = FakeSession(dumps=[NfcDump(present=False)])
    reader = make_reader(session)
    started(reader)
    try:
        assert not reader.dump().present
        assert session.dumps == 3  # DUMP_ATTEMPTS
    finally:
        reader.stop()


def test_dump_goes_through_the_reader_thread(driver_installed):
    session = FakeSession()
    reader = make_reader(session)
    started(reader)
    try:
        dump = reader.dump()
        assert (dump.present, dump.hex) == (True, "deadbeef")
        assert session.dumps == 1
    finally:
        reader.stop()


def test_dump_without_a_link_answers_instead_of_blocking(driver_installed):
    reader = NfcReader(
        port="/dev/ttyACM0",
        poll_interval=0.01,
        retry_interval=0.01,
        session_factory=lambda p: FakeSession(),
        port_resolver=lambda: None,
    )
    reader.start()
    try:
        assert not wait_until(reader.is_connected, timeout=0.2)
        assert reader.dump().error == "NOT_CONNECTED"
    finally:
        reader.stop()


def test_erase_is_served_by_the_reader_thread(driver_installed):
    session = FakeSession()
    reader = make_reader(session)
    started(reader)
    try:
        assert reader.erase(NfcEraseRequest()).success
        assert reader.erase(NfcEraseRequest(full=True)).success
        assert session.erases == [False, True]
    finally:
        reader.stop()


def test_erase_waits_for_a_tag_like_a_write_does(driver_installed):
    # Same posture as a write: the caller posts the erase, then presents the
    # tag. NO_TAG is not yet an answer.
    session = FakeSession(
        erases=[
            NfcWriteResult(success=False, error="NO_TAG"),
            NfcWriteResult(success=True),
        ]
    )
    reader = make_reader(session)
    started(reader)
    try:
        assert reader.erase(NfcEraseRequest()).success
        assert len(session.erases) == 2
    finally:
        reader.stop()


def test_erase_on_a_locked_tag_is_not_retried(driver_installed):
    session = FakeSession(erases=[NfcWriteResult(success=False, error="LOCKED")])
    reader = make_reader(session)
    started(reader)
    try:
        assert reader.erase(NfcEraseRequest()).error == "LOCKED"
        assert len(session.erases) == 1
    finally:
        reader.stop()


def test_erase_without_a_link_answers_instead_of_blocking(driver_installed):
    reader = NfcReader(
        port="/dev/ttyACM0",
        poll_interval=0.01,
        retry_interval=0.01,
        session_factory=lambda p: FakeSession(),
        port_resolver=lambda: None,
    )
    reader.start()
    try:
        assert not wait_until(reader.is_connected, timeout=0.2)
        assert reader.erase(NfcEraseRequest()).error == "NOT_CONNECTED"
    finally:
        reader.stop()


# --------------------------------------------------------------- the snapshot


class FakeInfo:
    """Stands in for winnie_nfc's TagInfo."""

    def __init__(self, name="NTAG215", ndef_bytes=496, read_only=False):
        self.name = name
        self.ndef_bytes = ndef_bytes
        self.read_only = read_only


_DEFAULT_INFO = object()  # so that info=None can mean "unidentified tag"


def reading(
    uid=b"\x04\xa1",
    info=_DEFAULT_INFO,
    records=None,
    data=b"",
    error=None,
):
    """Stands in for winnie_nfc's TagRead."""
    return SimpleNamespace(
        uid=uid,
        info=FakeInfo() if info is _DEFAULT_INFO else info,
        records=records,
        data=data,
        error=error,
    )


def test_snapshot_of_a_readable_tag():
    tag = _snapshot(
        reading(
            records=[
                {"type": "uri", "value": "https://pollen-robotics.com"},
            ],
            data=b"\x03\x10stuff",
        )
    )
    assert tag.present and tag.readable and not tag.blank
    assert tag.uid == "04A1"
    assert tag.model == "NTAG215"
    assert tag.capacity == 496
    assert tag.writable
    assert tag.content == "https://pollen-robotics.com"
    assert tag.records[0].kind == "uri"
    assert tag.raw_hex is None  # NDEF was decoded, raw bytes add nothing


def test_snapshot_of_a_factory_blank_tag():
    tag = _snapshot(reading(records=None, data=bytes(504)))
    assert tag.blank and tag.readable
    assert tag.content is None and tag.raw_hex is None


def test_snapshot_of_an_erased_tag_is_blank():
    # What an erase leaves behind: an NDEF message with zero records. The tag
    # is formatted and rewritable, and it must read as blank — reporting
    # "not blank, no content" would be a state nobody can act on.
    tag = _snapshot(reading(records=[], data=b"\x03\x00\xfe\x00"))
    assert tag.blank
    assert tag.readable and tag.content is None and tag.records == []
    assert tag.raw_hex is None, "an empty formatted tag has no bytes to show"


def test_snapshot_of_a_tag_carrying_a_bare_code_is_not_blank():
    # The real case met on the hardware: ASCII written straight into page 4,
    # no NDEF structure. Reporting it as blank would drop the only content.
    tag = _snapshot(reading(records=None, data=b"FC7FB644" + bytes(496)))
    assert not tag.blank
    assert tag.readable
    assert tag.raw_hex is not None
    assert bytes.fromhex(tag.raw_hex).startswith(b"FC7FB644")


def test_snapshot_of_an_unreadable_tag_keeps_the_uid():
    tag = _snapshot(reading(info=None, records=None, error="unusable tag: NAK"))
    assert tag.present and tag.uid == "04A1"
    assert not tag.readable and not tag.writable
    assert tag.error is not None
    assert tag.model is None and tag.capacity is None


def test_snapshot_of_a_locked_tag_is_not_writable():
    tag = _snapshot(reading(info=FakeInfo(read_only=True), records=[]))
    assert not tag.writable


def test_snapshot_reports_records_it_cannot_decode():
    tag = _snapshot(
        reading(
            records=[
                {
                    "type": "other",
                    "tnf": 2,
                    "type_name": "text/vcard",
                    "payload": b"BEGIN",
                }
            ]
        )
    )
    record = tag.records[0]
    assert record.kind == "other"
    assert record.type_name == "text/vcard"
    assert bytes.fromhex(record.data_hex) == b"BEGIN"
    # Nothing to show as content, but the tag is not blank either.
    assert tag.content is None and not tag.blank


# ------------------------------------------------------------ write requests


def test_a_write_request_takes_exactly_one_of_text_or_uri():
    assert NfcWriteRequest(text="a").text == "a"
    assert NfcWriteRequest(uri="https://a").uri == "https://a"
    for kwargs in ({}, {"text": "a", "uri": "https://a"}, {"text": ""}):
        with pytest.raises(ValueError):
            NfcWriteRequest(**kwargs)


def test_an_absurdly_long_write_is_refused_up_front():
    # The tag's own capacity is the real limit and the driver enforces it; this
    # only keeps the absurd from reaching the reader thread.
    with pytest.raises(ValueError):
        NfcWriteRequest(text="x" * 10_000)


# ------------------------------------------- the driver session's error map


@pytest.fixture
def fake_ntag_reader(monkeypatch):
    """Replace winnie_nfc's NTagReader so the session can be driven from a test.

    Skipped when the optional driver is not installed — this is the one part
    of the service that genuinely needs it, since what is under test is how
    the driver's own exceptions are translated.
    """
    pytest.importorskip("winnie_nfc")
    import winnie_nfc.reader as driver

    class FakeNTagReader:
        raises = None

        erased = []

        def __init__(self, port):
            self.port = port
            self.written = []

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def chip_version(self):
            return 0x1A

        def _write(self, value):
            if FakeNTagReader.raises is not None:
                raise FakeNTagReader.raises
            self.written.append(value)

        def write_text(self, text, lang="fr"):
            self._write(text)

        def write_uri(self, uri):
            self._write(uri)

        def erase(self, full=False):
            if FakeNTagReader.raises is not None:
                raise FakeNTagReader.raises
            FakeNTagReader.erased.append(full)
            return 1

    monkeypatch.setattr(driver, "NTagReader", FakeNTagReader)
    yield FakeNTagReader
    FakeNTagReader.raises = None
    FakeNTagReader.erased = []


def driver_exceptions():
    """The driver failures a write can hit, and the code each must produce."""
    from winnie_nfc import ndef, type2
    from winnie_nfc.core import CollisionError, NoTagError, TagError

    return [
        (NoTagError("nothing there"), "NO_TAG"),
        (CollisionError("two tags"), "COLLISION"),
        (type2.MessageTooLongError("too long"), "TOO_LONG"),
        (type2.TagLockedError("read only"), "LOCKED"),
        (type2.UnknownModelError("unknown"), "UNKNOWN_TAG"),
        (type2.WriteRefusedError("NAK"), "WRITE_REFUSED"),
        (ndef.NdefError("beyond any capacity"), "TOO_LONG"),
        (type2.Type2Error("odd"), "WRITE_ERROR"),
        # A tag that answers the SELECT but no Type 2 command — a MIFARE
        # Classic. Uncaught, this looked like a dead serial link and made the
        # reader thread drop a healthy one.
        (TagError("not a Type 2 tag"), "WRITE_ERROR"),
    ]


def test_every_driver_failure_maps_to_a_write_code(fake_ntag_reader):
    from reachy_mini.nfc.reader import Clrc663Session

    for exception, expected in driver_exceptions():
        session = Clrc663Session("/dev/null")
        fake_ntag_reader.raises = exception
        result = session.write("badge", None)
        assert not result.success
        assert result.error == expected, f"{type(exception).__name__}"
        assert result.detail == str(exception)


def test_a_transport_failure_is_left_to_the_reader_thread(fake_ntag_reader):
    # The one failure that must escape: the link is gone, and only the thread
    # can reopen it.
    from winnie_nfc.transport import TransportError

    from reachy_mini.nfc.reader import Clrc663Session

    session = Clrc663Session("/dev/null")
    fake_ntag_reader.raises = TransportError("port closed")
    with pytest.raises(TransportError):
        session.write("badge", None)


def test_erase_shares_the_write_error_map(fake_ntag_reader):
    from winnie_nfc import type2

    from reachy_mini.nfc.reader import Clrc663Session

    session = Clrc663Session("/dev/null")
    fake_ntag_reader.raises = type2.TagLockedError("read only")
    assert session.erase(full=False).error == "LOCKED"

    fake_ntag_reader.raises = None
    assert session.erase(full=True).success
    assert fake_ntag_reader.erased == [True]


def test_a_successful_write_reports_success(fake_ntag_reader):
    from reachy_mini.nfc.reader import Clrc663Session

    session = Clrc663Session("/dev/null")
    assert session.write("badge", None).success
    assert session.write(None, "https://pollen-robotics.com").success
    assert session.chip_version == "0x1A"
