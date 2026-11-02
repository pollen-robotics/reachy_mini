"""Tests for adopting the login a daemon before 1.12 left behind."""

import time
from pathlib import Path

from reachy_mini.apps.sources import hf_auth


def _hf_home() -> Path:
    # The autouse conftest fixture points both files at a fresh temp dir.
    return Path(hf_auth.HF_TOKEN_PATH).parent


def _write_legacy(token: str, stored_tokens: str | None = None) -> None:
    _hf_home().mkdir(parents=True, exist_ok=True)
    Path(hf_auth.HF_TOKEN_PATH).write_text(token + "\n", encoding="utf-8")
    if stored_tokens is not None:
        Path(hf_auth.HF_STORED_TOKENS_PATH).write_text(stored_tokens, encoding="utf-8")


def test_adopts_a_redirect_oauth_login() -> None:
    """Up to 1.11 a redirect login was only the bare token file."""
    _write_legacy("legacy-token")

    assert hf_auth.adopt_legacy_login() is True
    assert hf_auth.get_hf_token() == "legacy-token"
    # The shared files stay: they may still be someone's CLI login.
    assert Path(hf_auth.HF_TOKEN_PATH).exists()


def test_adopts_a_device_code_login_with_its_refresh_token() -> None:
    """A device-code login keeps refreshing after the update."""
    expires_at = int(time.time()) + 3600
    _write_legacy(
        "device-token",
        "[other]\nhf_token = someone-elses\n\n"
        f"[oauth-alice]\nhf_token = device-token\nrefresh_token = r%1\nexpires_at = {expires_at}\n",
    )

    assert hf_auth.adopt_legacy_login() is True
    stored = hf_auth._read_store()
    assert stored.access_token == "device-token"
    # interpolation=None: a `%` in a token is literal.
    assert stored.refresh_token == "r%1"
    assert stored.expires_at == expires_at


def test_never_undoes_a_sign_out() -> None:
    """Once the daemon has a record of its own, the legacy files are ignored."""
    _write_legacy("legacy-token")
    assert hf_auth.delete_hf_token() is True

    assert hf_auth.adopt_legacy_login() is False
    assert hf_auth.get_hf_token() is None


def test_does_nothing_without_a_legacy_login() -> None:
    """A robot that was never signed in stays signed out, without a record."""
    assert hf_auth.adopt_legacy_login() is False
    assert not hf_auth._store_path().exists()

    _write_legacy("  ")
    assert hf_auth.adopt_legacy_login() is False
    assert not hf_auth._store_path().exists()


def test_a_corrupt_stored_tokens_file_still_adopts_the_token() -> None:
    """Only the refresh metadata is lost, the robot stays signed in."""
    _write_legacy("legacy-token", "not an ini file\n[unterminated")

    assert hf_auth.adopt_legacy_login() is True
    stored = hf_auth._read_store()
    assert stored.access_token == "legacy-token"
    assert stored.refresh_token is None
