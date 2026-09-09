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

import asyncio
import logging
import socket
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import aiohttp
import httpx
import pytest
import uvicorn
from huggingface_hub import constants
from huggingface_hub.utils import _auth, _oauth_device

from reachy_mini.apps.sources import hf_auth
from reachy_mini.daemon.app import main
from reachy_mini.daemon.app.main import configure_root_logging


@pytest.fixture
def clean_root() -> Iterator[None]:
    """Root logger back to Python defaults, restored afterwards."""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    loggers = [
        logging.getLogger(name)
        for name in (
            "uvicorn",
            "uvicorn.error",
            "uvicorn.access",
            "huggingface_hub.utils._auth",
        )
    ]
    saved_configs = [
        (logger.filters[:], logger.handlers[:], logger.level, logger.propagate)
        for logger in loggers
    ]
    for logger in loggers:
        logger.filters.clear()
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    yield
    for handler in root.handlers:
        if handler not in saved_handlers:
            handler.close()
    root.handlers.clear()
    root.handlers.extend(saved_handlers)
    root.setLevel(saved_level)
    for logger, (filters, handlers, level, propagate) in zip(
        loggers, saved_configs, strict=True
    ):
        logger.filters[:] = filters
        logger.handlers[:] = handlers
        logger.setLevel(level)
        logger.propagate = propagate


def test_unconfigured_root_drops_info(clean_root, capsys):
    """Without configuration, INFO is lost. This is what the fix prevents."""
    logging.getLogger("reachy_mini.some.module").info("startup check result")

    assert capsys.readouterr().err == ""


def test_configure_sends_info_to_stderr(clean_root, capsys):
    """Systemd captures stderr, so INFO must reach it to land in the journal."""
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


@pytest.mark.asyncio
@pytest.mark.parametrize("http", ["h11", "httptools"])
async def test_oauth_access_logs_redact_only_callback_queries(
    clean_root: None,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    http: str,
) -> None:
    """Real HTTP requests redact the registered callback, including its redirect."""
    monkeypatch.setattr(main, "Daemon", MagicMock())
    monkeypatch.setattr(main, "AppManager", MagicMock())
    app = main.create_app(main.Args(autostart=False, no_media=True))
    callback = str(app.url_path_for("oauth_callback"))
    ordinary = "/api/hf-auth/oauth/configured"
    logfile = tmp_path / "daemon.log"
    configure_root_logging("INFO", str(logfile))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(
            uvicorn.Config(app, http=http, log_config=None, lifespan="off")
        )
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            async with asyncio.timeout(5):
                while not server.started:
                    await asyncio.sleep(0.01)
                async with aiohttp.ClientSession() as client:
                    for path, status in (
                        (
                            callback
                            + "?error=access_denied&code=secret-code&state=secret-state",
                            200,
                        ),
                        (callback + "/?code=secret-code&state=secret-state", 307),
                        (ordinary + "?next=" + callback, 200),
                        (ordinary + "?next=/health-check", 200),
                        ("/api/hf-auth/relay-status", 200),
                    ):
                        async with client.get(
                            f"http://127.0.0.1:{sock.getsockname()[1]}{path}",
                            allow_redirects=False,
                        ) as response:
                            await response.read()
                            assert response.status == status
        finally:
            server.should_exit = True
            await task

    for output in (capsys.readouterr().err, logfile.read_text()):
        assert f'{callback}?<redacted> HTTP/1.1" 200' in output
        assert f'{callback}/?<redacted> HTTP/1.1" 307' in output
        assert ordinary + "?next=" + callback in output
        assert ordinary + "?next=/health-check" in output
        assert "/api/hf-auth/relay-status" not in output
        assert "secret-code" not in output
        assert "secret-state" not in output


@pytest.mark.parametrize("error_code", ["invalid_grant", "server_error"])
def test_hub_refresh_preserves_credentials_without_logging_provider_text(
    clean_root: None,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error_code: str,
) -> None:
    """The real Hub refresh failure keeps the cached token and emits a safe warning."""
    token_path = tmp_path / "token"
    stored_path = tmp_path / "stored_tokens"
    token_path.write_text("hf_old_token")
    stored = "[audit]\nhf_token = hf_old_token\nrefresh_token = hf_refresh_token\nexpires_at = 1\n"
    stored_path.write_text(stored)
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
        json={"error": error_code, "error_description": "provider-secret-marker"},
    )
    monkeypatch.setattr(_oauth_device, "get_session", lambda: provider)
    logfile = tmp_path / "daemon.log"
    configure_root_logging("INFO", str(logfile))

    assert hf_auth.get_hf_token() == "hf_old_token"
    assert hf_auth.get_hf_token() == "hf_old_token"
    provider.post.assert_called_once()
    assert token_path.read_text() == "hf_old_token"
    assert stored_path.read_text() == stored
    for output in (capsys.readouterr().err, logfile.read_text()):
        assert "Hugging Face credential lookup or refresh failed" in output
        assert "provider-secret-marker" not in output
        assert "hf_old_token" not in output
        assert "hf_refresh_token" not in output
