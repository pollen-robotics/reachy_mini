"""Unit tests for the wireless_version update helpers."""

import logging
import shlex
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import requests
import semver

from reachy_mini.utils.wireless_version import update, update_available, utils

# --- build_install_command ---


def test_build_pypi_version_uv(monkeypatch):
    """PyPI pinned version via uv, with upgrade flag."""
    monkeypatch.setattr(utils, "_check_uv_available", lambda: True)
    cmd, env = utils.build_install_command("wireless-version", version="1.2.3", upgrade=True)
    tokens = shlex.split(cmd)
    assert tokens[:3] == ["uv", "pip", "install"]
    assert "reachy-mini[wireless-version]==1.2.3" in tokens
    assert "--upgrade" in tokens
    assert env == {}


def test_build_pypi_prerelease_pip_fallback(monkeypatch):
    """PyPI latest pre-release falls back to bare pip when uv is missing."""
    monkeypatch.setattr(utils, "_check_uv_available", lambda: False)
    cmd, env = utils.build_install_command("wireless-version", pre_release=True)
    tokens = shlex.split(cmd)
    assert tokens[:2] == ["pip", "install"]
    assert "reachy-mini[wireless-version]" in tokens
    assert "--pre" in tokens
    assert "--upgrade" not in tokens
    assert env == {}


def test_build_verbose_flag(monkeypatch):
    """Verbose adds -vvv to the base command."""
    monkeypatch.setattr(utils, "_check_uv_available", lambda: True)
    cmd, _ = utils.build_install_command("gstreamer", verbose=True)
    assert "-vvv" in shlex.split(cmd)


def test_build_external_venv_uv(monkeypatch):
    """External venv install targets --python when uv is available."""
    monkeypatch.setattr(utils, "_check_uv_available", lambda: True)
    py = Path("/venvs/apps_venv/bin/python")
    cmd, _ = utils.build_install_command("", python=py, upgrade=True)
    tokens = shlex.split(cmd)
    assert tokens[:5] == ["uv", "pip", "install", "--python", str(py)]


def test_build_external_venv_pip_fallback(monkeypatch):
    """External venv without uv uses that venv's own pip binary."""
    monkeypatch.setattr(utils, "_check_uv_available", lambda: False)
    py = Path("/venvs/apps_venv/bin/python")
    cmd, _ = utils.build_install_command("", python=py)
    tokens = shlex.split(cmd)
    assert tokens[:2] == [str(py.parent / "pip"), "install"]


def _split_git_ref_cmd(cmd):
    """Split `step1 && ( step2 || step3 )` into its three steps."""
    step1, rest = cmd.split(" && ", 1)
    assert rest.startswith("( ") and rest.endswith(" )")
    step2, step3 = rest[2:-2].split(" || ", 1)
    return step1, step2, step3


def test_build_git_ref_uv(monkeypatch):
    """Git ref install chains three steps and sets the LFS env var (uv)."""
    monkeypatch.setattr(utils, "_check_uv_available", lambda: True)
    cmd, env = utils.build_install_command("wireless-version", git_ref="main")
    assert env == {"GIT_LFS_SKIP_SMUDGE": "1"}

    step1, step2, step3 = _split_git_ref_cmd(cmd)

    t1 = shlex.split(step1)
    assert "--force-reinstall" in t1 and "--no-deps" in t1 and "--no-cache-dir" in t1
    assert any("git+https://github.com/pollen-robotics/reachy_mini.git@main" in tok for tok in t1)

    assert shlex.split(step2) == ["uv", "pip", "check"]

    t3 = shlex.split(step3)
    assert "--upgrade" in t3
    assert "--upgrade-strategy" not in t3  # uv path omits the strategy flag


def test_build_git_ref_pip_fallback(monkeypatch):
    """Git ref via pip appends the only-if-needed upgrade strategy."""
    monkeypatch.setattr(utils, "_check_uv_available", lambda: False)
    cmd, _ = utils.build_install_command("wireless-version", git_ref="dev")
    _, step2, step3 = _split_git_ref_cmd(cmd)
    assert shlex.split(step2) == ["pip", "check"]
    t3 = shlex.split(step3)
    assert "--upgrade-strategy" in t3 and "only-if-needed" in t3


def test_git_ref_grouping_shell_precedence():
    """Regression: with the grouped form, a failed step1 fails the whole chain."""
    assert subprocess.run("false && true || true", shell=True).returncode == 0  # old form, masked
    assert subprocess.run("false && ( true || true )", shell=True).returncode != 0  # grouped, honest


# --- call_logger_wrapper ---


@pytest.mark.asyncio
async def test_call_logger_wrapper_success():
    """A zero exit code returns without raising."""
    await utils.call_logger_wrapper("true", logging.getLogger("test"))


@pytest.mark.asyncio
async def test_call_logger_wrapper_nonzero_raises():
    """Regression: a non-zero exit code must raise, not pass silently."""
    with pytest.raises(RuntimeError, match="exit code 7"):
        await utils.call_logger_wrapper("exit 7", logging.getLogger("test"))


@pytest.mark.asyncio
async def test_call_logger_wrapper_tolerated_returncode():
    """A code listed in ok_returncodes does not raise (the restart's 143/-15)."""
    # 143 = exit status relayed by the shell.
    await utils.call_logger_wrapper("exit 143", logging.getLogger("test"), ok_returncodes=(0, -15, 143))
    # -15 = process killed by SIGTERM (asyncio reports the negated signal), the
    # real self-inflicted signal seen inside the daemon's cgroup.
    await utils.call_logger_wrapper("kill -TERM $$", logging.getLogger("test"), ok_returncodes=(0, -15, 143))
    # A code outside the list still raises.
    with pytest.raises(RuntimeError, match="exit code 5"):
        await utils.call_logger_wrapper("exit 5", logging.getLogger("test"), ok_returncodes=(0, -15, 143))


def test_restart_argv_matches_sudoers_whitelist():
    """The restart argv is whitelisted verbatim in /etc/sudoers.d/010-pollen-reachy.

    Any extra flag (e.g. --no-block) stops sudo matching the NOPASSWD rule, so the
    restart fails with "a password is required" and the update never applies.
    """
    src = Path(update.__file__).read_text()
    assert '"sudo systemctl restart reachy-mini-daemon"' in src


# --- _semver_version ---


def test_semver_plain():
    """Plain semver strings parse directly."""
    assert update_available._semver_version("1.2.3") == semver.Version.parse("1.2.3")


def test_semver_rc():
    """PyPI-style rc suffix is normalized to a semver pre-release."""
    v = update_available._semver_version("1.2.3rc4")
    assert (v.major, v.minor, v.patch) == (1, 2, 3)
    assert v.prerelease == "rc.4"


def test_semver_invalid_raises():
    """Unparseable strings raise ValueError."""
    with pytest.raises(ValueError):
        update_available._semver_version("not-a-version")


def test_semver_unparseable_three_parts_raises():
    """A 3-part string that still won't parse (no rc suffix) raises ValueError."""
    with pytest.raises(ValueError):
        update_available._semver_version("1.2.x")


def test_semver_pep440_dev():
    """PEP 440 `.devN` normalizes to a semver pre-release.

    The package's own version on main is `X.Y.Z.dev0` (pyproject), so any
    from-source / from-git-ref install reports a `.dev0` string to
    `get_local_version` -> `_semver_version`.
    """
    v = update_available._semver_version("1.10.0.dev0")
    assert (v.major, v.minor, v.patch) == (1, 10, 0)
    assert v.prerelease == "dev.0"


# --- get_local_version ---


def test_get_local_version_pep440_dev(monkeypatch):
    """Regression: a `.dev0` install must not crash the version check.

    `get_local_version` feeds `importlib.metadata.version(...)` straight into
    `_semver_version`. On a from-ref / from-source install that string is the
    `1.10.0.dev0` marker, and `_semver_version` raises ValueError today —
    breaking `/available`, the `/update` gating, and the daemon update flow.
    """
    monkeypatch.setattr(update_available, "version", lambda _pkg: "1.10.0.dev0")
    v = update_available.get_local_version("reachy_mini")
    assert (v.major, v.minor, v.patch) == (1, 10, 0)
    assert v.prerelease == "dev.0"


# --- get_pypi_version ---


class _FakeResponse:
    """Minimal stand-in for a requests Response."""

    def __init__(self, payload, error=None):
        self._payload = payload
        self._error = error

    def raise_for_status(self):
        if self._error:
            raise self._error

    def json(self):
        return self._payload


def test_get_pypi_version_stable(monkeypatch):
    """Stable lookup returns info.version."""
    payload = {"info": {"version": "2.0.0"}, "releases": {}}
    monkeypatch.setattr(update_available.requests, "get", lambda *a, **k: _FakeResponse(payload))
    assert update_available.get_pypi_version("reachy-mini", pre_release=False) == semver.Version.parse("2.0.0")


def test_get_pypi_version_prerelease(monkeypatch):
    """Pre-release lookup prefers a newer trailing release entry."""
    payload = {"info": {"version": "2.0.0"}, "releases": {"1.9.0": [], "2.1.0rc1": []}}
    monkeypatch.setattr(update_available.requests, "get", lambda *a, **k: _FakeResponse(payload))
    v = update_available.get_pypi_version("reachy-mini", pre_release=True)
    assert (v.major, v.minor, v.patch) == (2, 1, 0)


def test_get_pypi_version_prerelease_pep440_info_version(monkeypatch):
    """Regression: pre-release compare must not crash on a PEP 440 info.version.

    `pre_version > version` compared a parsed Version against the RAW
    info.version string; semver coerces the RHS with strict parse, so a
    non-semver info.version (e.g. an rc release) raised ValueError.
    """
    payload = {"info": {"version": "2.0.0rc1"}, "releases": {"1.9.0": [], "2.0.0rc2": []}}
    monkeypatch.setattr(
        update_available.requests, "get", lambda *a, **k: _FakeResponse(payload)
    )
    v = update_available.get_pypi_version("reachy-mini", pre_release=True)
    assert (v.major, v.minor, v.patch) == (2, 0, 0)  # rc2 is newer -> returned
    assert v.prerelease == "rc.2"


def test_get_pypi_version_http_error(monkeypatch):
    """HTTP errors propagate out of get_pypi_version."""
    resp = _FakeResponse({}, error=requests.HTTPError("boom"))
    monkeypatch.setattr(update_available.requests, "get", lambda *a, **k: resp)
    with pytest.raises(requests.HTTPError):
        update_available.get_pypi_version("reachy-mini", pre_release=False)


# --- get_install_source ---


class _FakeDist:
    """Fake importlib.metadata Distribution."""

    def __init__(self, text, raise_missing=False):
        self._text = text
        self._raise = raise_missing

    def read_text(self, name):
        if self._raise:
            raise FileNotFoundError(name)
        return self._text


def _patch_source(monkeypatch, dist, ver="1.0.0"):
    """Wire up distribution/version for get_install_source."""
    monkeypatch.setattr(update_available, "distribution", lambda name: dist)
    monkeypatch.setattr(update_available, "version", lambda name: ver)


def test_install_source_pypi_no_direct_url(monkeypatch):
    """No direct_url.json text means a PyPI install."""
    _patch_source(monkeypatch, _FakeDist(None))
    result = update_available.get_install_source("reachy-mini")
    assert result == {"version": "1.0.0", "source": "pypi"}


def test_install_source_editable(monkeypatch):
    """dir_info.editable marks an editable install."""
    text = '{"dir_info": {"editable": true}, "url": "file:///x"}'
    _patch_source(monkeypatch, _FakeDist(text))
    assert update_available.get_install_source("reachy-mini")["source"] == "editable"


def test_install_source_git(monkeypatch):
    """vcs_info yields git source with ref and short commit."""
    text = '{"vcs_info": {"requested_revision": "main", "commit_id": "abcdef1234567890"}}'
    _patch_source(monkeypatch, _FakeDist(text))
    result = update_available.get_install_source("reachy-mini")
    assert result["source"] == "git"
    assert result["git_ref"] == "main"
    assert result["commit"] == "abcdef12"


def test_install_source_file_not_found(monkeypatch):
    """A missing direct_url.json falls back to PyPI."""
    _patch_source(monkeypatch, _FakeDist(None, raise_missing=True))
    assert update_available.get_install_source("reachy-mini")["source"] == "pypi"


# --- is_update_available ---


def test_is_update_available_true(monkeypatch):
    """Newer PyPI version reports an available update."""
    monkeypatch.setattr(update_available, "get_pypi_version", lambda p, pr: semver.Version.parse("2.0.0"))
    monkeypatch.setattr(update_available, "get_local_version", lambda p: semver.Version.parse("1.0.0"))
    assert update_available.is_update_available("reachy-mini", False) is True


def test_is_update_available_false(monkeypatch):
    """Equal versions report no update."""
    monkeypatch.setattr(update_available, "get_pypi_version", lambda p, pr: semver.Version.parse("1.0.0"))
    monkeypatch.setattr(update_available, "get_local_version", lambda p: semver.Version.parse("1.0.0"))
    assert update_available.is_update_available("reachy-mini", False) is False


# --- update_reachy_mini ---


@pytest.mark.asyncio
async def test_update_with_apps_venv(monkeypatch):
    """Update installs daemon + apps venv, then restarts the daemon."""
    calls = AsyncMock()
    monkeypatch.setattr(update, "call_logger_wrapper", calls)
    monkeypatch.setattr(update.Path, "exists", lambda self: True)

    await update.update_reachy_mini(logging.getLogger("test"))

    assert calls.call_count == 3
    assert calls.await_args_list[-1].args[0] == "sudo systemctl restart reachy-mini-daemon"


@pytest.mark.asyncio
async def test_update_without_apps_venv(monkeypatch):
    """Missing apps venv skips the second install step."""
    calls = AsyncMock()
    monkeypatch.setattr(update, "call_logger_wrapper", calls)
    monkeypatch.setattr(update.Path, "exists", lambda self: False)

    await update.update_reachy_mini(logging.getLogger("test"), pre_release=True)

    assert calls.call_count == 2
    assert calls.await_args_list[-1].args[0] == "sudo systemctl restart reachy-mini-daemon"


@pytest.mark.asyncio
async def test_update_daemon_install_failure_aborts_before_restart(monkeypatch):
    """A failed daemon venv install propagates and never restarts the daemon."""
    calls = AsyncMock(side_effect=RuntimeError("pip install failed"))
    monkeypatch.setattr(update, "call_logger_wrapper", calls)
    monkeypatch.setattr(update.Path, "exists", lambda self: True)

    with pytest.raises(RuntimeError, match="pip install failed"):
        await update.update_reachy_mini(logging.getLogger("test"))

    # Only the daemon venv install ran: no apps_venv step, no restart.
    assert calls.call_count == 1


@pytest.mark.asyncio
async def test_update_apps_venv_failure_still_restarts(monkeypatch):
    """An apps_venv failure is non-fatal: the restart still applies the update."""
    commands: list[str] = []

    async def fake_call(cmd, logger, env=None, ok_returncodes=(0,)):
        commands.append(cmd)
        if len(commands) == 2:  # second call = apps_venv install
            raise RuntimeError("apps_venv install failed")

    monkeypatch.setattr(update, "call_logger_wrapper", fake_call)
    monkeypatch.setattr(update.Path, "exists", lambda self: True)

    await update.update_reachy_mini(logging.getLogger("test"))

    assert len(commands) == 3
    assert commands[-1] == "sudo systemctl restart reachy-mini-daemon"


@pytest.mark.asyncio
async def test_update_restart_failure_propagates(monkeypatch):
    """A failed restart command surfaces as a failed job, not a silent pass."""
    commands: list[str] = []

    async def fake_call(cmd, logger, env=None, ok_returncodes=(0,)):
        commands.append(cmd)
        if cmd.startswith("sudo systemctl"):
            raise RuntimeError("systemctl failed")

    monkeypatch.setattr(update, "call_logger_wrapper", fake_call)
    monkeypatch.setattr(update.Path, "exists", lambda self: False)

    with pytest.raises(RuntimeError, match="systemctl failed"):
        await update.update_reachy_mini(logging.getLogger("test"))
