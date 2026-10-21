"""Tests for the daemon's anonymous telemetry (issue #1452)."""

import asyncio
import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_mini.daemon import startup_app_config
from reachy_mini.daemon import telemetry as tm
from reachy_mini.daemon.app.routers import telemetry as telemetry_router
from reachy_mini.io.protocol import GetTelemetryCmd, SetTelemetryCmd

FORBIDDEN_KEYS = {"robot_name", "hostname", "hardware_id", "ip", "ssid", "username"}


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[MagicMock]:
    """Point config and state at a temp dir, clear opt-out env vars, stub the network."""
    monkeypatch.setattr(tm, "_state_path", lambda: tmp_path / "telemetry_state.json")
    monkeypatch.setattr(
        startup_app_config, "_config_path", lambda: tmp_path / "daemon_config.json"
    )
    monkeypatch.setattr(tm, "OS_IMAGE_VERSION_PATH", tmp_path / "VERSION.txt")
    for var in ("REACHY_MINI_TELEMETRY", "DO_NOT_TRACK", "HF_HUB_DISABLE_TELEMETRY"):
        monkeypatch.delenv(var, raising=False)
    post = MagicMock(return_value=MagicMock(status_code=200))
    monkeypatch.setattr(tm.requests, "post", post)
    yield post


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Return the user-editable daemon config file."""
    return tmp_path / "daemon_config.json"


@pytest.fixture
def client() -> Iterator[tm.Telemetry]:
    """Fresh, stopped-after-use instance."""
    instance = tm.Telemetry()
    yield instance
    instance._halt(flush=False)


def _start(t: tm.Telemetry, form_factor: str = "wireless") -> None:
    t.start(form_factor=form_factor, desktop_app_daemon=False)


def _events(t: tm.Telemetry) -> list[str]:
    return [e["event"] for e in t._queue]


def test_parse_client_header() -> None:
    """Valid headers parse, unknown values collapse to other, garbage is dropped."""
    assert tm.parse_client_header("desktop/0.9.5 (macos)") == {
        "client_type": "desktop",
        "client_version": "0.9.5",
        "client_os": "macos",
    }
    assert tm.parse_client_header("toaster/1.0 (beos)") == {
        "client_type": "other",
        "client_version": "1.0",
        "client_os": "other",
    }
    assert tm.parse_client_header("<script>/1 (x)") is None
    assert tm.parse_client_header("desktop/" + "9" * 40) is None


def test_app_id_only_for_provably_public_apps() -> None:
    """Private or unknown apps never leak their Space id."""
    assert tm.app_id_for({"id": "pollen/app", "private": False}) == "pollen/app"
    assert tm.app_id_for({"id": "me/secret", "private": True}) == "private"
    assert tm.app_id_for({"id": "me/unknown"}) == "private"
    assert tm.app_id_for({}) == "private"


def test_wireless_reports_os_image_version(tmp_path: Path) -> None:
    """Wireless reads the image tag from VERSION.txt; others report the host OS."""
    (tmp_path / "VERSION.txt").write_text(
        "ReachyMiniOS: dryrun-develop-0ecedbe\nCreated on: 2026-09-08\n"
    )
    assert tm.os_fields("wireless") == {
        "os_type": "reachy_mini_os",
        "os_version": "dryrun-develop-0ecedbe",
    }
    assert tm.os_fields("lite")["os_type"] in {"linux", "macos", "windows"}


def test_wireless_without_version_file_falls_back_to_host() -> None:
    """A missing VERSION.txt (e.g. a dev board) reports the host OS instead."""
    assert tm.os_fields("wireless")["os_type"] != "reachy_mini_os"


def test_on_by_default_and_key_written(client: tm.Telemetry, config_file: Path) -> None:
    """First start activates and writes the key so users can find and edit it."""
    _start(client)
    assert client.active
    assert json.loads(config_file.read_text())["telemetry_enabled"] is True


def test_start_sends_anonymous_envelope(client: tm.Telemetry) -> None:
    """daemon_started carries the envelope and no identifying fields."""
    _start(client)

    assert _events(client) == ["daemon_started"]
    event = client._queue[0]
    props = event["properties"]
    assert props["form_factor"] == "wireless"
    assert props["event_source"] == "daemon"
    assert props["$geoip_disable"] is True
    assert "time_to_ready_ms" in props
    assert not FORBIDDEN_KEYS & props.keys()
    assert event["distinct_id"] == tm._read_state()["robot_id"]


def test_hand_edited_false_disables_and_forgets(
    client: tm.Telemetry, config_file: Path
) -> None:
    """Setting the key to false in the file turns telemetry off and drops the id."""
    tm._write_state({"robot_id": "old"})
    config_file.write_text(json.dumps({"telemetry_enabled": False, "startup_app": "x"}))
    _start(client)
    assert not client.active
    assert tm._read_state() == {}
    assert json.loads(config_file.read_text())["startup_app"] == "x"


@pytest.mark.parametrize(
    "content", ['{"telemetry_enabled": fals', '{"telemetry_enabled": "no"}']
)
def test_broken_config_counts_as_off_and_is_left_alone(
    client: tm.Telemetry, config_file: Path, content: str
) -> None:
    """A typo while opting out never enables telemetry nor rewrites the file."""
    config_file.write_text(content)
    _start(client)
    assert not client.active
    assert config_file.read_text() == content


def test_version_change_reports_update_success(client: tm.Telemetry) -> None:
    """A different version than at the previous boot means an update succeeded."""
    tm._write_state({"robot_id": "r", "last_version": "0.0.1"})
    _start(client)

    assert _events(client) == ["daemon_started", "update_result"]
    update = client._queue[1]["properties"]
    assert update["from_version"] == "0.0.1"
    assert update["success"] is True


def test_unclean_previous_exit(client: tm.Telemetry) -> None:
    """A run that never called stop() is reported as an unclean exit."""
    tm._write_state({"robot_id": "r", "running": True})
    _start(client)
    assert client._queue[0]["properties"]["previous_exit_clean"] is False


def test_disabling_forgets_robot_id(client: tm.Telemetry, config_file: Path) -> None:
    """Opting out drops pending events and the id; opting back in gets a new id."""
    _start(client)
    first_id = client._robot_id

    client.set_enabled(False)
    assert not client.active
    assert not client._queue
    assert tm._read_state() == {}
    assert json.loads(config_file.read_text())["telemetry_enabled"] is False

    client.set_enabled(True)
    assert client.active
    assert client._robot_id != first_id


@pytest.mark.parametrize(
    "var,value",
    [
        ("DO_NOT_TRACK", "1"),
        ("HF_HUB_DISABLE_TELEMETRY", "true"),
        ("REACHY_MINI_TELEMETRY", "0"),
    ],
)
def test_env_overrides_user_choice(
    client: tm.Telemetry, monkeypatch: pytest.MonkeyPatch, var: str, value: str
) -> None:
    """An opt-out environment variable wins over the stored choice."""
    monkeypatch.setenv(var, value)
    _start(client)
    assert not client.active


def test_client_connected_once_per_client(client: tm.Telemetry) -> None:
    """Each distinct client is reported once; garbage headers are ignored."""
    _start(client)
    client.note_client("desktop/1.0 (windows)")
    client.note_client("desktop/1.0 (windows)")
    client.note_client("nonsense")
    assert _events(client).count("client_connected") == 1


def test_opt_out_during_failing_send_drops_the_batch(
    client: tm.Telemetry, isolated: MagicMock
) -> None:
    """Events in flight when the user opts out are never retried."""
    _start(client)

    def opt_out_then_fail(*args: object, **kwargs: object) -> None:
        client.set_enabled(False)
        raise requests.ConnectionError

    isolated.side_effect = opt_out_then_fail
    client._flush(timeout=1)
    assert not client._queue
    client._flush(timeout=1)
    assert isolated.call_count == 1


def test_opt_in_again_does_not_revive_old_sender(client: tm.Telemetry) -> None:
    """Each sender thread has its own stop event, so only one keeps running."""
    _start(client)
    old_stop = client._stop
    client.set_enabled(False)
    client.set_enabled(True)
    assert old_stop.is_set()
    assert client._stop is not old_stop and not client._stop.is_set()



def test_opt_out_does_not_wait_for_a_stuck_send(
    client: tm.Telemetry, isolated: MagicMock
) -> None:
    """Opting out returns at once even while a send hangs (slow network)."""
    release = threading.Event()
    sending = threading.Event()

    def hang(*args: object, **kwargs: object) -> MagicMock:
        sending.set()
        release.wait(5)
        return MagicMock(status_code=200)

    isolated.side_effect = hang
    _start(client)
    client._wake.set()
    assert sending.wait(2)

    started = time.monotonic()
    client.set_enabled(False)
    assert time.monotonic() - started < 0.5
    assert not client.active
    release.set()

def test_flush_posts_batch_and_requeues_when_offline(
    client: tm.Telemetry, isolated: MagicMock
) -> None:
    """Events go to PostHog in one batch; a network error keeps them for later."""
    _start(client)

    isolated.side_effect = requests.ConnectionError
    client._flush(timeout=1)
    assert _events(client) == ["daemon_started"]

    isolated.side_effect = None
    client._flush(timeout=1)
    payload = isolated.call_args.kwargs["json"]
    assert payload["api_key"] == tm.POSTHOG_KEY
    assert [e["event"] for e in payload["batch"]] == ["daemon_started"]
    assert not client._queue


def test_router_get_and_put(
    client: tm.Telemetry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The API reads and toggles the choice in the config file."""
    monkeypatch.setattr(telemetry_router, "telemetry", client)
    app = FastAPI()
    app.include_router(telemetry_router.router, prefix="/api")
    http = TestClient(app)

    assert http.get("/api/telemetry").json() == {
        "enabled": True,
        "disabled_by_env": False,
    }
    assert (
        http.put("/api/telemetry", json={"enabled": False}).json()["enabled"] is False
    )
    assert startup_app_config.get_telemetry_enabled() is False


@pytest.mark.asyncio
async def test_data_channel_get_and_set(
    client: tm.Telemetry, monkeypatch: pytest.MonkeyPatch, sim_backend: Any
) -> None:
    """The data channel reads and toggles the choice like the HTTP API."""
    monkeypatch.setattr(tm, "telemetry", client)
    responses: list[dict[str, Any]] = []

    sim_backend.process_command(GetTelemetryCmd(), send_response=responses.append)
    sim_backend.process_command(
        SetTelemetryCmd(enabled=False), send_response=responses.append
    )
    for _ in range(100):
        if len(responses) == 2:
            break
        await asyncio.sleep(0.01)

    assert responses == [
        {
            "command": "get_telemetry",
            "status": "ok",
            "enabled": True,
            "disabled_by_env": False,
        },
        {
            "command": "set_telemetry",
            "status": "ok",
            "enabled": False,
            "disabled_by_env": False,
        },
    ]
    assert startup_app_config.get_telemetry_enabled() is False
