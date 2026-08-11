"""Tests for the daemon's root logging configuration.

The wireless launcher starts the daemon with neither --log-file nor any
prior logging setup, so anything logged before the handlers are installed
goes to a root logger sitting at its WARNING default with no handlers:
INFO is dropped outright and WARNING falls back to logging.lastResort
without a formatter. The startup checks run in exactly that window, which
is why configure_root_logging has to happen before them.

configure_root_logging is called inside each test body rather than in a
fixture: logging.StreamHandler binds sys.stderr at construction time, and
pytest swaps sys.stderr between the fixture and the call phase.
"""

import logging

import pytest

from reachy_mini.daemon.app.main import configure_root_logging


@pytest.fixture
def clean_root():
    """Root logger back to Python defaults, restored afterwards."""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    yield
    root.handlers.clear()
    root.handlers.extend(saved_handlers)
    root.setLevel(saved_level)


def test_unconfigured_root_drops_info(clean_root, capsys):
    """Without configuration, INFO is lost. This is what the fix prevents."""
    logging.getLogger("reachy_mini.some.module").info("startup check result")

    assert capsys.readouterr().err == ""


def test_configure_sends_info_to_stderr(clean_root, capsys):
    """systemd captures stderr, so INFO must reach it to land in the journal."""
    configure_root_logging("INFO")
    logging.getLogger("reachy_mini.some.module").info("startup check result")

    err = capsys.readouterr().err
    assert "startup check result" in err
    assert "reachy_mini.some.module" in err, "records must carry the module name"
    assert "INFO" in err, "records must carry the level"


def test_log_file_is_added_alongside_stderr_not_instead(clean_root, capsys, tmp_path):
    """A --log-file must not cost us the journal."""
    logfile = tmp_path / "daemon.log"
    configure_root_logging("INFO", str(logfile))
    logging.getLogger("reachy_mini.some.module").warning("something odd")
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert "something odd" in capsys.readouterr().err
    assert "something odd" in logfile.read_text()


def test_configure_is_idempotent(clean_root, capsys):
    """Calling it twice must not double every log line."""
    configure_root_logging("INFO")
    configure_root_logging("INFO")
    logging.getLogger("reachy_mini.some.module").info("once")

    assert capsys.readouterr().err.count("once") == 1


def test_level_is_honoured(clean_root, capsys):
    """A quieter level still filters, so --log-level keeps working."""
    configure_root_logging("WARNING")
    log = logging.getLogger("reachy_mini.some.module")
    log.info("chatty")
    log.warning("important")

    err = capsys.readouterr().err
    assert "chatty" not in err
    assert "important" in err
