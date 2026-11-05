"""Unit tests for the optional NFC reader accessory.

Serial ports and reader sessions are injected: no hardware needed.
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


def comport(device, product=None):
    """A fake entry of serial.tools.list_ports.comports()."""
    return SimpleNamespace(device=device, product=product)


# --------------------------------------------------------------- port lookup


@pytest.mark.parametrize(
    "product, expected",
    [
        ("Winnie NFC 1.2", (True, "1.2")),
        ("  winnie nfc v2 ", (True, "v2")),
        ("Winnie NFC", (True, None)),
        ("USB Single Serial", (False, None)),
        ("", (False, None)),
        (None, (False, None)),
    ],
)
def test_parse_product(product, expected):
    assert ports.parse_product(product) == expected


def test_find_nfc_board_goes_by_the_product_string():
    # Only the product string identifies the board; it carries the version.
    found = ports.find_nfc_board(
        comports=[
            comport("/dev/ttyACM0", product="USB Single Serial"),  # motors
            comport("/dev/ttyUSB0", product="ReSpeaker 4 Mic Array"),
            comport("/dev/ttyACM1", product="Winnie NFC 1.0"),
        ]
    )
    assert found == ports.NfcBoard(port="/dev/ttyACM1", version="1.0")


def test_find_nfc_board_honours_exclusions():
    found = ports.find_nfc_board(
        comports=[
            comport("/dev/ttyACM0", product="Winnie NFC 1.0"),
            comport("/dev/ttyACM1", product="Winnie NFC 2.0"),
        ],
        exclude=["/dev/ttyACM0"],
    )
    assert found == ports.NfcBoard(port="/dev/ttyACM1", version="2.0")


def test_find_nfc_board_returns_none_without_the_product_string():
    assert ports.find_nfc_board(comports=[comport("/dev/ttyACM0")]) is None


def test_describe_port_reads_the_version_of_a_configured_port():
    comports = [comport("/dev/ttyACM1", product="Winnie NFC 1.3")]
    assert ports.describe_port("/dev/ttyACM1", comports=comports).version == "1.3"
    assert ports.describe_port("/dev/ttyACM9", comports=comports) == ports.NfcBoard(
        port="/dev/ttyACM9", version=None
    )


# ------------------------------------------------------------- fake session


class FakeSession:
    """A reader session whose answers the test dictates.

    ``tags`` overrides ``tag``: one snapshot per poll, the last one repeating.
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
        self.forgets = 0
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

    def forget(self):
        self.forgets += 1

    def close(self):
        self.closed = True


@pytest.fixture
def driver_installed(monkeypatch):
    """Pretend the optional winnie_nfc driver is installed."""
    monkeypatch.setattr("reachy_mini.nfc.reader.driver_available", lambda: True)


@pytest.fixture
def no_board(monkeypatch):
    """Make auto-detection find no reader board."""
    monkeypatch.setattr("reachy_mini.nfc.reader.find_nfc_board", lambda **kw: None)


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


def test_reader_reports_the_board_version(driver_installed, monkeypatch):
    monkeypatch.setattr(
        "reachy_mini.nfc.reader.find_nfc_board",
        lambda **kw: ports.NfcBoard(port="/dev/ttyACM1", version="1.2"),
    )
    reader = make_reader(FakeSession(), port="auto")
    started(reader)
    try:
        status = reader.get_status()
        assert (status.port, status.board_version) == ("/dev/ttyACM1", "1.2")
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


def test_the_reader_can_be_switched_off_and_on_again(driver_installed):
    session = FakeSession(tag=NfcTag(present=True, uid="04"))
    reader = make_reader(session)
    assert not reader.get_status().enabled
    started(reader)
    try:
        assert reader.get_status().enabled
        reader.stop()
        status = reader.get_status()
        assert not status.enabled
        assert not status.connected
        assert status.error == "NFC reader disabled"
        assert status.error_code == "DISABLED"
        started(reader)
        assert reader.get_status().enabled
        assert wait_until(lambda: reader.get_tag().present)
    finally:
        reader.stop()


def test_stopping_answers_a_pending_write(driver_installed):
    # A pending write is answered when the reader stops.
    session = FakeSession(
        write_results=[NfcWriteResult(success=False, error="NO_TAG")] * 1000
    )
    reader = make_reader(session)
    started(reader)
    results = []
    writer = threading.Thread(
        target=lambda: results.append(
            reader.write(NfcWriteRequest(text="x"), timeout=30)
        )
    )
    writer.start()
    assert wait_until(lambda: session.write_calls)
    started_at = time.monotonic()
    reader.stop()
    writer.join(timeout=5)
    assert not writer.is_alive()
    assert time.monotonic() - started_at < 5
    assert not results[0].success


@pytest.fixture
def daemon_config(tmp_path, monkeypatch):
    """Point the daemon config file at a temporary path."""
    from reachy_mini.daemon import startup_app_config

    path = tmp_path / "daemon_config.json"
    monkeypatch.setattr(startup_app_config, "_config_path", lambda: path)
    return path


def test_the_switch_is_remembered_and_on_by_default(daemon_config):
    from reachy_mini.daemon import startup_app_config as config

    assert config.get_nfc_enabled()
    assert config.set_nfc_enabled(False)
    assert not config.get_nfc_enabled()
    assert config.set_nfc_enabled(True)
    assert config.get_nfc_enabled()
    daemon_config.write_text("not json")
    assert config.get_nfc_enabled()


def test_the_switch_lives_with_the_other_daemon_settings(daemon_config):
    import json

    from reachy_mini.daemon import startup_app_config as config

    config.set_startup_app("my_app")
    config.set_nfc_enabled(False)
    assert json.loads(daemon_config.read_text()) == {
        "startup_app": "my_app",
        "nfc_enabled": False,
    }


def test_a_switch_that_cannot_be_saved_still_applies(
    driver_installed, tmp_path, monkeypatch
):
    from reachy_mini.daemon import startup_app_config
    from reachy_mini.daemon import nfc_switch

    # mkdir fails, as on a read-only filesystem.
    (tmp_path / "blocker").write_text("")
    monkeypatch.setattr(
        startup_app_config,
        "_config_path",
        lambda: tmp_path / "blocker" / "daemon_config.json",
    )
    reader = make_reader(FakeSession())
    started(reader)
    try:
        status = nfc_switch.switch_reader(reader, False)
        assert not status.enabled
        assert status.setting_saved is False
        assert reader.get_status().setting_saved is None
    finally:
        reader.stop()


def test_concurrent_switches_leave_the_saved_and_live_state_in_agreement(
    driver_installed, monkeypatch
):
    # A switch must not run between saving and applying another one.
    from reachy_mini.daemon import nfc_switch, startup_app_config

    saved = []
    saving_on = threading.Event()

    def save(enabled):
        saved.append(enabled)
        if enabled:
            saving_on.set()
            time.sleep(0.1)  # widen the window between saving and applying
        return True

    monkeypatch.setattr(startup_app_config, "set_nfc_enabled", save)
    reader = make_reader(FakeSession())
    try:
        on = threading.Thread(target=nfc_switch.switch_reader, args=(reader, True))
        off = threading.Thread(target=nfc_switch.switch_reader, args=(reader, False))
        on.start()
        assert saving_on.wait(2)  # "on" is saved first
        off.start()
        on.join(timeout=5)
        off.join(timeout=5)
        assert saved == [True, False]
        assert not reader.is_enabled()
    finally:
        reader.stop()


def test_concurrent_config_writes_keep_every_key(daemon_config):
    # Concurrent writes from several threads keep every key.
    import json

    from reachy_mini.daemon import startup_app_config as config

    def write(key):
        for i in range(20):
            config._update(key, i)

    threads = [threading.Thread(target=write, args=(f"k{n}",)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert json.loads(daemon_config.read_text()) == {f"k{n}": 19 for n in range(8)}


def test_a_saved_switch_says_so(driver_installed, daemon_config):
    from reachy_mini.daemon import startup_app_config
    from reachy_mini.daemon import nfc_switch

    reader = make_reader(FakeSession())
    try:
        status = nfc_switch.switch_reader(reader, True)
        assert status.enabled
        assert status.setting_saved is True
        assert startup_app_config.get_nfc_enabled()
    finally:
        reader.stop()


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
    # NO_TAG is retried until a tag is presented.
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
    # Definitive refusals are not retried.
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


def test_write_without_a_link_answers_instead_of_blocking(driver_installed, no_board):
    reader = NfcReader(
        poll_interval=0.01,
        retry_interval=0.01,
        session_factory=lambda p: FakeSession(),
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
    )
    reader.start()
    try:
        assert wait_until(lambda: len(sessions) >= 2)
        assert sessions[0].closed, "the dead session must be released"
        assert wait_until(lambda: reader.is_connected())
    finally:
        reader.stop()


def test_the_driver_lookup_is_done_once(monkeypatch):
    import builtins

    from reachy_mini.nfc.reader import driver_available

    driver_available.cache_clear()
    attempts = []
    real_import = builtins.__import__

    def tracking_import(name, *args, **kwargs):
        if name == "winnie_nfc":
            attempts.append(name)
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", tracking_import)
    try:
        assert not driver_available()
        assert not driver_available()
        assert attempts == ["winnie_nfc"]
    finally:
        driver_available.cache_clear()


def test_a_broken_driver_install_reads_as_unavailable(monkeypatch):
    # Any import failure, not only ImportError, reads as a missing driver.
    import builtins

    from reachy_mini.nfc.reader import driver_available

    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "winnie_nfc":
            raise OSError("libusb not found")
        return real_import(name, *args, **kwargs)

    driver_available.cache_clear()
    monkeypatch.setattr(builtins, "__import__", broken_import)
    try:
        assert not driver_available()
    finally:
        driver_available.cache_clear()


def test_a_missing_driver_disables_the_reader_without_failing(monkeypatch):
    monkeypatch.setattr("reachy_mini.nfc.reader.driver_available", lambda: False)
    reader = make_reader(FakeSession())
    reader.start()  # must not raise, must not start a thread
    try:
        assert not reader.is_connected()
        status = reader.get_status()
        assert not status.driver_available
        assert status.error is not None and "winnie_nfc" in status.error
        assert status.error_code == "DRIVER_MISSING"
        assert reader.write(NfcWriteRequest(text="x")).error == "DRIVER_MISSING"
        assert threading.active_count() >= 1
    finally:
        reader.stop()


def test_a_single_missed_detection_is_not_a_removal(driver_installed):
    # One missed detection is not a removal.
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


def test_dump_without_a_link_answers_instead_of_blocking(driver_installed, no_board):
    reader = NfcReader(
        poll_interval=0.01,
        retry_interval=0.01,
        session_factory=lambda p: FakeSession(),
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
    # NO_TAG is retried until a tag is presented, as for a write.
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


def test_erase_without_a_link_answers_instead_of_blocking(driver_installed, no_board):
    reader = NfcReader(
        poll_interval=0.01,
        retry_interval=0.01,
        session_factory=lambda p: FakeSession(),
    )
    reader.start()
    try:
        assert not wait_until(reader.is_connected, timeout=0.2)
        assert reader.erase(NfcEraseRequest()).error == "NOT_CONNECTED"
    finally:
        reader.stop()


def test_an_explicit_port_that_is_the_motor_port_is_refused(driver_installed):
    # An explicit port that is the motor controller's is never opened.
    opened = []

    def factory(port):
        opened.append(port)
        return FakeSession()

    reader = NfcReader(
        port="/dev/ttyACM0",
        exclude_ports=("/dev/ttyACM0",),
        poll_interval=0.01,
        retry_interval=0.01,
        session_factory=factory,
    )
    reader.start()
    try:
        assert wait_until(lambda: reader.get_status().error_code == "PORT_CONFLICT")
        assert "motor controller" in (reader.get_status().error or "")
        assert opened == []
        assert not reader.is_connected()
    finally:
        reader.stop()


def test_a_missing_board_has_its_own_code(driver_installed, no_board):
    reader = NfcReader(poll_interval=0.01, retry_interval=0.01)
    reader.start()
    try:
        assert wait_until(lambda: reader.get_status().error_code == "NO_BOARD")
    finally:
        reader.stop()
    assert reader.get_status().error_code == "DISABLED"


def test_a_board_that_cannot_be_opened_is_a_link_error(driver_installed):
    def factory(port):
        raise OSError("permission denied")

    reader = NfcReader(
        port="/dev/ttyACM0",
        poll_interval=0.01,
        retry_interval=0.01,
        session_factory=factory,
    )
    reader.start()
    try:
        assert wait_until(lambda: reader.get_status().error_code == "LINK_ERROR")
        assert reader.get_status().error == "permission denied"
    finally:
        reader.stop()


def test_a_connected_reader_reports_no_error_code(driver_installed):
    reader = make_reader(FakeSession())
    started(reader)
    try:
        status = reader.get_status()
        assert status.error is None
        assert status.error_code is None
    finally:
        reader.stop()


def test_a_job_dequeued_after_a_stop_never_reaches_the_tag():
    # A job dequeued after stop() is answered without touching the tag.
    from reachy_mini.nfc.reader import _WriteJob

    session = FakeSession()
    reader = make_reader(session)
    reader._should_stop.set()
    job = _WriteJob(lambda s: s.write("x", None), timeout=5)
    reader._serve_job(session, job)
    assert session.write_calls == []
    assert job.outcome is not None
    assert job.outcome.error == "NOT_CONNECTED"


class BlockingSession(FakeSession):
    """A session whose erase (or poll) holds the chip until released."""

    def __init__(self, block_erase=False, block_polls=False, **kwargs):
        super().__init__(**kwargs)
        self.release = threading.Event()
        self.erasing = threading.Event()
        self.polling = threading.Event()  # set once a poll is blocked
        self.block_erase = block_erase
        self.block_polls = block_polls

    def read_tag(self):
        if self.block_polls:
            self.polling.set()
            self.release.wait(5)
        return super().read_tag()

    def erase(self, full):
        if self.block_erase:
            self.erasing.set()
            self.release.wait(5)
        return super().erase(full)


def _reader_threads():
    return [t for t in threading.enumerate() if t.name == "nfc-reader"]


def test_a_restart_during_a_long_job_never_runs_two_threads(driver_installed):
    # Restarting during a long job reuses the running thread.
    session = BlockingSession(block_erase=True)
    opened = []

    def factory(port):
        opened.append(port)
        return session

    reader = NfcReader(
        port="/dev/ttyACM0",
        poll_interval=0.01,
        retry_interval=0.01,
        session_factory=factory,
    )
    started(reader)
    eraser = threading.Thread(
        target=lambda: reader.erase(NfcEraseRequest(full=True), timeout=10)
    )
    eraser.start()
    try:
        assert session.erasing.wait(2)
        reader.stop(timeout=0.05)
        reader.start()
        assert len(_reader_threads()) == 1
        session.release.set()
        eraser.join(timeout=5)
        assert reader.get_status().enabled
        assert wait_until(reader.is_connected)
        assert len(_reader_threads()) == 1
        assert opened == ["/dev/ttyACM0"]
    finally:
        session.release.set()
        reader.stop()
    assert wait_until(lambda: not _reader_threads())


def test_a_stop_during_a_long_job_still_ends_the_thread(driver_installed):
    session = BlockingSession(block_erase=True)
    reader = make_reader(session)
    started(reader)
    eraser = threading.Thread(
        target=lambda: reader.erase(NfcEraseRequest(full=True), timeout=10)
    )
    eraser.start()
    assert session.erasing.wait(2)
    reader.stop(timeout=0.05)
    session.release.set()
    eraser.join(timeout=5)
    assert wait_until(lambda: not _reader_threads())
    assert not reader.is_connected()


def test_a_timed_out_write_never_reaches_the_tag(driver_installed):
    # A write that timed out while queued never runs.
    session = BlockingSession()
    reader = make_reader(session)
    started(reader)
    try:
        session.block_polls = True
        assert session.polling.wait(2)  # the thread is inside a blocked poll
        result = reader.write(NfcWriteRequest(text="x"), timeout=0.01)
        assert result.error == "TIMEOUT"
        session.block_polls = False
        polls = session.polls
        session.release.set()
        # Polling resumes only once the withdrawn job has been dequeued.
        assert wait_until(lambda: session.polls >= polls + 2)
        assert session.write_calls == []
    finally:
        session.release.set()
        reader.stop()


def test_a_write_whose_attempt_is_running_waits_for_its_outcome(
    driver_installed, monkeypatch
):
    # An attempt running when the first wait ends: its real outcome is returned.
    from reachy_mini.nfc.reader import _Job

    first_wait_over = threading.Event()
    cancel = _Job.cancel

    def spy_cancel(job):
        first_wait_over.set()  # cancel() is only called once a wait timed out
        return cancel(job)

    monkeypatch.setattr(_Job, "cancel", spy_cancel)
    session = BlockingSession(block_erase=True)
    reader = make_reader(session)
    started(reader)
    results = []
    eraser = threading.Thread(
        target=lambda: results.append(reader.erase(NfcEraseRequest(), timeout=0.01))
    )
    try:
        eraser.start()
        assert session.erasing.wait(2)
        assert first_wait_over.wait(5)
        session.release.set()
        eraser.join(timeout=5)
        assert results[0].success
    finally:
        session.release.set()
        reader.stop()


def test_a_write_still_on_the_tag_when_the_caller_gives_up_is_uncertain(
    driver_installed,
):
    # An attempt still running after both waits: UNCERTAIN.
    session = BlockingSession(block_erase=True)
    reader = make_reader(session)
    started(reader)
    results = []
    eraser = threading.Thread(
        target=lambda: results.append(reader.erase(NfcEraseRequest(), timeout=0.01))
    )
    try:
        eraser.start()
        assert session.erasing.wait(2)
        eraser.join(timeout=5)  # both waits run out while the erase blocks
        assert results[0].error == "UNCERTAIN"
    finally:
        session.release.set()
        reader.stop()


def test_a_job_queued_when_the_link_drops_is_answered_at_once(driver_installed):
    # A job queued when the link drops is answered LINK_LOST at once and never runs.
    session = BlockingSession()
    opened = []

    def factory(port):
        opened.append(port)
        if len(opened) > 1:
            raise OSError("no board")
        return session

    reader = NfcReader(
        port="/dev/ttyACM0",
        poll_interval=0.01,
        retry_interval=0.01,
        session_factory=factory,
    )
    started(reader)
    results = []
    writer = threading.Thread(
        target=lambda: results.append(
            reader.write(NfcWriteRequest(text="x"), timeout=30)
        )
    )
    try:
        session.block_polls = True
        assert session.polling.wait(2)  # the thread is inside a blocked poll
        writer.start()
        assert wait_until(lambda: reader._jobs.qsize() == 1)  # the write is queued
        session.fail_on_poll = 1
        session.release.set()
        writer.join(timeout=5)
        assert not writer.is_alive()
        assert results[0].error == "LINK_LOST"
        assert session.write_calls == []
    finally:
        session.release.set()
        reader.stop()


def test_a_job_queued_after_the_thread_exited_is_answered_at_once(
    driver_installed, monkeypatch
):
    # stop() completes between the connection check and the queueing: the job
    # is answered NOT_CONNECTED at once instead of waiting for its timeout.
    session = FakeSession()
    reader = make_reader(session)
    started(reader)
    reader.stop()
    assert not _reader_threads()
    monkeypatch.setattr(reader, "is_connected", lambda: True)  # checked before stop
    begun = time.monotonic()
    result = reader.write(NfcWriteRequest(text="x"), timeout=30)
    assert result.error == "NOT_CONNECTED"
    assert time.monotonic() - begun < 1.0
    assert session.write_calls == []


def test_a_job_queued_behind_a_long_one_answers_busy(driver_installed, monkeypatch):
    monkeypatch.setattr("reachy_mini.nfc.reader.JOB_LOCK_TIMEOUT", 0.05)
    session = BlockingSession(block_erase=True)
    reader = make_reader(session)
    started(reader)
    eraser = threading.Thread(
        target=lambda: reader.erase(NfcEraseRequest(full=True), timeout=10)
    )
    try:
        eraser.start()
        assert session.erasing.wait(2)
        assert reader.write(NfcWriteRequest(text="x")).error == "BUSY"
        assert reader.dump().error == "BUSY"
    finally:
        session.release.set()
        eraser.join(timeout=5)
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
    # An empty NDEF message (an erased tag) reads as blank.
    tag = _snapshot(reading(records=[], data=b"\x03\x00\xfe\x00"))
    assert tag.blank
    assert tag.readable and tag.content is None and tag.records == []
    assert tag.raw_hex is None, "an empty formatted tag has no bytes to show"


def test_snapshot_of_a_tag_carrying_a_bare_code_is_not_blank():
    # Raw bytes in page 4 without NDEF are reported in raw_hex, not as blank.
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
    assert tag.content is None and not tag.blank


# ------------------------------------------------------------ write requests


def test_a_write_request_takes_exactly_one_of_text_or_uri():
    assert NfcWriteRequest(text="a").text == "a"
    assert NfcWriteRequest(uri="https://a").uri == "https://a"
    for kwargs in ({}, {"text": "a", "uri": "https://a"}, {"text": ""}):
        with pytest.raises(ValueError):
            NfcWriteRequest(**kwargs)


def test_an_absurdly_long_write_is_refused_up_front():
    # Upper bound only; the tag's capacity is checked by the driver.
    with pytest.raises(ValueError):
        NfcWriteRequest(text="x" * 10_000)


# ------------------------------------------- the driver session's error map


@pytest.fixture
def fake_ntag_reader(monkeypatch):
    """Replace winnie_nfc's NTagReader (skipped when the driver is not installed)."""
    pytest.importorskip("winnie_nfc")
    import winnie_nfc.reader as driver

    class FakeNTagReader:
        raises = None

        erased = []

        def __init__(self, port):
            self.port = port
            self.written = []
            self.fields_off = 0

        def __enter__(self):
            return self

        def field_off(self):
            self.fields_off += 1

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
        # Not a Type 2 tag (MIFARE Classic).
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
    # A transport failure propagates so the reader thread reconnects.
    from winnie_nfc.transport import TransportError

    from reachy_mini.nfc.reader import Clrc663Session

    session = Clrc663Session("/dev/null")
    fake_ntag_reader.raises = TransportError("port closed")
    with pytest.raises(TransportError):
        session.write("badge", None)


def test_an_unexpected_write_failure_is_a_write_error_not_a_link_loss(
    fake_ntag_reader,
):
    from reachy_mini.nfc.reader import Clrc663Session

    for exception in (
        ValueError("bad uri"),
        UnicodeEncodeError("ascii", "é", 0, 1, "x"),
    ):
        session = Clrc663Session("/dev/null")
        fake_ntag_reader.raises = exception
        result = session.write("badge", None)
        assert result.error == "WRITE_ERROR", type(exception).__name__


def test_a_serial_failure_still_reaches_the_reader_thread(fake_ntag_reader):
    # SerialException is an OSError: the link is gone.
    from reachy_mini.nfc.reader import Clrc663Session

    session = Clrc663Session("/dev/null")
    fake_ntag_reader.raises = OSError("device disconnected")
    with pytest.raises(OSError):
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


def test_a_failing_field_off_does_not_hide_a_landed_write(
    fake_ntag_reader, monkeypatch
):
    # A failing field_off does not change the write's result.
    from reachy_mini.nfc.reader import Clrc663Session

    session = Clrc663Session("/dev/null")

    def broken_field_off():
        raise OSError("port gone")

    monkeypatch.setattr(session._reader, "field_off", broken_field_off)
    result = session.write("badge", None)
    assert result.success
    assert session._reader.written == ["badge"]


def test_every_session_call_ends_with_the_field_off(fake_ntag_reader):
    from winnie_nfc.core import NoTagError

    from reachy_mini.nfc.reader import Clrc663Session

    session = Clrc663Session("/dev/null")
    session._reader.uid = lambda: b"\x04\xa1"
    session._reader.read_tag = lambda: reading(records=[])
    calls = [
        session.read_tag,
        session.read_tag,  # served from the cache, after a detection
        session.dump,
        lambda: session.write("badge", None),
        lambda: session.erase(full=False),
    ]
    for call in calls:
        session._reader.fields_off = 0
        call()
        assert session._reader.fields_off == 1

    fake_ntag_reader.raises = NoTagError("nothing there")
    session._reader.fields_off = 0
    assert not session.write("badge", None).success
    assert session._reader.fields_off == 1


def test_a_write_is_seen_without_lifting_the_tag(fake_ntag_reader):
    # A write keeps the UID: the cached content must be dropped.
    from reachy_mini.nfc.reader import Clrc663Session

    session = Clrc663Session("/dev/null")
    reads = []

    def read_tag():
        reads.append(1)
        return reading(records=[{"type": "text", "value": f"v{len(reads)}"}])

    session._reader.uid = lambda: b"\x04\xa1"
    session._reader.read_tag = read_tag

    assert session.read_tag().records[0].value == "v1"
    assert session.read_tag().records[0].value == "v1"  # cached
    assert session.write("v2", None).success
    assert session.read_tag().records[0].value == "v2"
    assert session.erase(full=False).success
    assert session.read_tag().records[0].value == "v3"


def _session_reading(results):
    """Build a session whose tag answers the UID and gives `results` in turn.

    Each item is a TagRead, or an exception for the UID detection to raise.
    """
    from reachy_mini.nfc.reader import Clrc663Session

    session = Clrc663Session("/dev/null")
    pending = list(results)
    session.reads = 0

    def uid():
        if pending and isinstance(pending[0], Exception):
            raise pending.pop(0)
        return b"\x04\xa1"

    def read_tag():
        session.reads += 1
        return pending.pop(0)

    session._reader.uid = uid
    session._reader.read_tag = read_tag
    return session


def test_a_failed_content_read_is_not_kept(fake_ntag_reader):
    # A failed content read is retried on the next polls.
    good = reading(records=[{"type": "text", "value": "hf_pirate"}])
    session = _session_reading([reading(error="cannot read: no answer"), good])

    first = session.read_tag()
    assert first.present and first.uid == "04A1"
    assert not first.readable and not first.blank

    assert session.read_tag().content == "hf_pirate"
    assert session.read_tag().content == "hf_pirate"  # now cached
    assert session.reads == 2


def test_a_tag_that_never_reads_stops_being_re_read(fake_ntag_reader):
    from reachy_mini.nfc.reader import CONTENT_READ_POLLS

    failed = reading(info=None, error="unusable tag: NAK")
    session = _session_reading([failed] * CONTENT_READ_POLLS)

    for _ in range(CONTENT_READ_POLLS + 3):
        tag = session.read_tag()
        assert tag.present and not tag.readable
    assert session.reads == CONTENT_READ_POLLS


def test_a_missed_detection_keeps_the_content_read(fake_ntag_reader):
    # The cached read survives missed detections.
    from winnie_nfc.core import NoTagError

    from reachy_mini.nfc.reader import ABSENCE_POLLS

    good = reading(records=[{"type": "text", "value": "hf_pirate"}])
    misses = [NoTagError("no answer")] * (ABSENCE_POLLS - 1)
    session = _session_reading([good, *misses, good])

    assert session.read_tag().content == "hf_pirate"
    for _ in misses:
        assert not session.read_tag().present
    assert session.read_tag().content == "hf_pirate"
    assert session.reads == 1


def test_forgetting_drops_the_content_read(fake_ntag_reader):
    from winnie_nfc.core import NoTagError

    first = reading(records=[{"type": "text", "value": "v1"}])
    second = reading(records=[{"type": "text", "value": "v2"}])
    session = _session_reading([first, NoTagError("no answer"), second])

    assert session.read_tag().content == "v1"
    assert not session.read_tag().present
    session.forget()
    assert session.read_tag().content == "v2"


def test_the_session_cache_follows_the_readers_absence_count(driver_installed):
    # The session forgets exactly when the reader confirms the absence.
    here = NfcTag(present=True, uid="04", readable=True)
    gone = NfcTag(present=False)
    session = FakeSession(tags=[here, gone, gone, gone, gone, gone, gone])
    reader = make_reader(session, absence_polls=5)
    reader._record_poll(here)
    for _ in range(4):
        assert not reader._record_poll(gone)
        assert reader.get_tag().present
    assert reader._record_poll(gone)
    assert not reader.get_tag().present
    assert not reader._record_poll(gone)  # forgotten once, not on every poll


def test_the_reader_thread_tells_the_session_to_forget(driver_installed):
    here = NfcTag(present=True, uid="04", readable=True)
    gone = NfcTag(present=False)
    session = FakeSession(tags=[here, gone])
    reader = make_reader(session, absence_polls=2)
    started(reader)
    try:
        assert wait_until(lambda: session.forgets == 1)
        assert not reader.get_tag().present
        polls = session.polls
        assert wait_until(lambda: session.polls >= polls + 3)
        assert session.forgets == 1
    finally:
        reader.stop()


def test_an_undecodable_tag_is_reported_not_taken_for_a_link_loss(
    fake_ntag_reader,
):
    # An undecodable record is reported in the snapshot, not as a link error.
    session = _session_reading([reading(records=[{"type": "text"}])])
    tag = session.read_tag()
    assert tag.present
    assert tag.uid == "04A1"
    assert not tag.readable
    assert tag.error is not None and "undecodable" in tag.error


def test_a_malformed_frame_at_detection_is_a_miss_not_a_tag(fake_ntag_reader):
    from winnie_nfc.iso14443a import MalformedResponseError

    good = reading(records=[{"type": "text", "value": "hf_pirate"}])
    session = _session_reading([good, MalformedResponseError("bad BCC"), good])

    assert session.read_tag().content == "hf_pirate"
    assert not session.read_tag().present
    assert session.read_tag().content == "hf_pirate"
    assert session.reads == 1


def test_a_collision_is_still_reported_as_a_present_tag(fake_ntag_reader):
    from winnie_nfc.core import CollisionError

    session = _session_reading([CollisionError("two tags")])
    tag = session.read_tag()
    assert tag.present and tag.error == "two tags"


def test_a_successful_write_reports_success(fake_ntag_reader):
    from reachy_mini.nfc.reader import Clrc663Session

    session = Clrc663Session("/dev/null")
    assert session.write("badge", None).success
    assert session.write(None, "https://pollen-robotics.com").success
    assert session.chip_version == "0x1A"
