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
from unittest.mock import MagicMock

import httpx
import pytest
from huggingface_hub import constants
from huggingface_hub.utils import _auth, _oauth_device

from reachy_mini.apps.sources import hf_auth
from reachy_mini.daemon.app import main
from reachy_mini.daemon.app.main import configure_root_logging


@pytest.fixture
def clean_root():
    """Root logger back to Python defaults, restored afterwards."""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    filtered_loggers = [
        logging.getLogger("uvicorn.access"),
        logging.getLogger("huggingface_hub.utils._auth"),
    ]
    saved_filters = [logger.filters[:] for logger in filtered_loggers]
    for logger in filtered_loggers:
        logger.filters.clear()
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    yield
    root.handlers.clear()
    root.handlers.extend(saved_handlers)
    root.setLevel(saved_level)
    for logger, filters in zip(filtered_loggers, saved_filters, strict=True):
        logger.filters[:] = filters


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


def _access_record(path: str) -> logging.LogRecord:
    return logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        0,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:54321", "GET", path, "1.1", 200),
        None,
    )


def test_access_log_filter_redacts_only_callback_queries() -> None:
    """Redact only callback query strings."""
    callback = _access_record("/api/hf-auth/oauth/callback/?code=secret&state=secret")
    ordinary = _access_record("/api/hf-auth/oauth/configured?next=/health-check")
    for record in (callback, ordinary):
        assert main.access_log_filter(record)

    assert callback.getMessage().endswith('callback/?<redacted> HTTP/1.1" 200')
    assert ordinary.getMessage().endswith('configured?next=/health-check HTTP/1.1" 200')


def test_hub_stored_tokens_parse_error_redacts_token(
    clean_root, capsys, monkeypatch, tmp_path
) -> None:
    """Keep stored token lines out of Hub parse errors."""
    token_path = tmp_path / "token"
    stored_path = tmp_path / "stored_tokens"
    token_path.write_text("hf_old_token")
    stored_path.write_text("hf_token = hf_old_token\n")
    monkeypatch.setattr(constants, "HF_TOKEN_PATH", str(token_path))
    monkeypatch.setattr(constants, "HF_STORED_TOKENS_PATH", str(stored_path))
    monkeypatch.setattr(_auth, "_OAUTH_REFRESH_CACHE", None)
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_OIDC_RESOURCE"):
        monkeypatch.delenv(name, raising=False)
    configure_root_logging("INFO")

    assert hf_auth.get_hf_token() == "hf_old_token"

    output = capsys.readouterr().err
    assert "Could not parse the stored Hugging Face tokens file" in output
    assert "hf_old_token" not in output


@pytest.mark.parametrize("error_code", ["invalid_grant", "server_error"])
def test_hub_refresh_log_redacts_provider_text(
    clean_root, capsys, monkeypatch, tmp_path, error_code
) -> None:
    """Redact provider text from Hub refresh warnings."""
    token_path = tmp_path / "token"
    stored_path = tmp_path / "stored_tokens"
    token_path.write_text("hf_old_token")
    stored_path.write_text(
        "[audit]\nhf_token = hf_old_token\n"
        "refresh_token = hf_refresh_token\nexpires_at = 1\n"
    )
    monkeypatch.setattr(constants, "HF_TOKEN_PATH", str(token_path))
    monkeypatch.setattr(constants, "HF_STORED_TOKENS_PATH", str(stored_path))
    monkeypatch.setattr(_auth, "_OAUTH_REFRESH_CACHE", None)
    monkeypatch.setattr(_auth, "_OAUTH_REFRESH_WARNED", False)
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_OIDC_RESOURCE"):
        monkeypatch.delenv(name, raising=False)
    provider = MagicMock()
    provider.post.return_value = httpx.Response(
        400,
        request=httpx.Request("POST", "https://huggingface.co/oauth/token"),
        json={
            "error": error_code,
            "error_description": "provider-secret-marker",
        },
    )
    monkeypatch.setattr(_oauth_device, "get_session", lambda: provider)
    configure_root_logging("INFO")

    assert hf_auth.get_hf_token() == "hf_old_token"

    output = capsys.readouterr().err
    assert "Hugging Face credential refresh failed" in output
    assert "provider-secret-marker" not in output
    assert "hf_old_token" not in output
    assert "hf_refresh_token" not in output
