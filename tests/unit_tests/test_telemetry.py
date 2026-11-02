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
from reachy_mini.io.protocol import ClientHelloCmd, GetTelemetryCmd, SetTelemetryCmd

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
        "client_type_raw": "toaster",
    }
    assert tm.parse_client_header("<script>/1 (x)") is None
    assert tm.parse_client_header("web/1.2.0") == {
        "client_type": "web",
        "client_version": "1.2.0",
    }
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


def _props(t: tm.Telemetry, event: str) -> list[dict[str, Any]]:
    return [e["properties"] for e in t._queue if e["event"] == event]


def test_unknown_client_type_keeps_its_raw_name() -> None:
    """An unknown client is still bucketed as other, but we learn its name."""
    assert tm.parse_client_header("toaster/1.0 (ios)") == {
        "client_type": "other",
        "client_version": "1.0",
        "client_os": "ios",
        "client_type_raw": "toaster",
    }
    assert "client_type_raw" not in tm.parse_client_header("mobile/1.0 (ios)")  # type: ignore[operator]


def test_hello_reports_client_and_js_app_lifecycle(client: tm.Telemetry) -> None:
    """A data-channel hello names the client; the peer leaving ends the JS app."""
    _start(client)
    client.note_hello("p1", "web/1.2.0", "pollen-robotics/emotions")
    client.peer_closed("p1")
    client.peer_closed("p1")  # second disconnect notice: no duplicate stop

    assert _props(client, "client_connected")[0]["client_type"] == "web"
    assert _props(client, "js_app_started") == [
        {**_props(client, "js_app_started")[0], "app_id": "pollen-robotics/emotions"}
    ]
    stopped = _props(client, "js_app_stopped")
    assert len(stopped) == 1 and stopped[0]["duration_s"] >= 0


@pytest.mark.parametrize(
    "app", ["private", "../../etc", "a" * 150, "no-slash", "x/y z", "john/<b>"]
)
def test_hello_never_reports_an_unvetted_app_id(client: tm.Telemetry, app: str) -> None:
    """Anything that isn't a plain owner/name id is sent as private."""
    _start(client)
    client.note_hello("p1", "web/1.2.0", app)
    assert _props(client, "js_app_started")[0]["app_id"] == "private"


def test_hello_without_app_tracks_usage_only(client: tm.Telemetry) -> None:
    """The mobile app itself is a client, not a JS app."""
    _start(client)
    client.note_hello("p1", "mobile/0.11.4 (android)", None)
    client.peer_closed("p1")
    assert not _props(client, "js_app_started")
    assert not _props(client, "js_app_stopped")
    assert _props(client, "client_connected")[0]["client_type"] == "mobile"


def test_repeated_hello_keeps_the_session(client: tm.Telemetry) -> None:
    """The same hello again is not a new run; a different app replaces it."""
    _start(client)
    for _ in range(3):
        client.note_hello("p1", "web/1.2.0", "pollen-robotics/emotions")
    assert len(_props(client, "js_app_started")) == 1
    assert not _props(client, "js_app_stopped")

    client.note_hello("p1", "web/1.2.0", "pollen-robotics/radio")
    assert [e["app_id"] for e in _props(client, "js_app_stopped")] == [
        "pollen-robotics/emotions"
    ]
    assert [e["app_id"] for e in _props(client, "js_app_started")] == [
        "pollen-robotics/emotions",
        "pollen-robotics/radio",
    ]


def test_garbage_hello_is_ignored(client: tm.Telemetry) -> None:
    """A malformed client string reports nothing and opens no session."""
    _start(client)
    client.note_hello("p1", "<script>", "pollen-robotics/emotions")
    assert _events(client) == ["daemon_started"]
    assert not client._sessions


def test_usage_tick_counts_anything_seen_in_the_window(client: tm.Telemetry) -> None:
    """A session that started and ended between two ticks still counts once."""
    _start(client)
    client._usage_tick()
    assert not _props(client, "usage_tick")  # idle robot: nothing sent

    client.session_begin("app:x", "python_app", "pollen-robotics/hello")
    client.note_hello("p1", "web/1.2.0", "private")
    client.session_end("app:x")
    client._usage_tick()
    client._usage_tick()  # p1 still connected
    client.peer_closed("p1")
    client._usage_tick()  # p1 was seen in this window
    client._usage_tick()  # nothing left

    ticks = _props(client, "usage_tick")
    assert [t["kinds"] for t in ticks] == [["python_app", "web"], ["web"], ["web"]]
    assert ticks[0]["apps"] == ["pollen-robotics/hello", "private"]
    assert ticks[0]["interval_s"] == int(tm.USAGE_TICK_S)


def test_usage_tick_hides_a_python_apps_own_sdk(client: tm.Telemetry) -> None:
    """A Python app talks through the SDK: that's one client, not two."""
    _start(client)
    client.session_begin("sdk:1", "python_sdk")
    client._usage_tick()
    client.session_begin("app:x", "python_app", "pollen-robotics/hello")
    client._usage_tick()
    assert [t["kinds"] for t in _props(client, "usage_tick")] == [
        ["python_sdk"],
        ["python_app"],
    ]


def test_sessions_are_capped(client: tm.Telemetry) -> None:
    """Peers whose disconnect never arrives can't grow memory without bound."""
    for i in range(200):
        client.session_begin(f"peer:{i}", "web")
    assert len(client._sessions) == tm._MAX_SESSIONS


@pytest.mark.parametrize(
    "dataset,move,expected",
    [
        (
            "pollen-robotics/reachy-mini-emotions-library",
            "happy1",
            {
                "dataset": "pollen-robotics/reachy-mini-emotions-library",
                "move": "happy1",
            },
        ),
        ("john/secret-moves", "happy1", {"dataset": "other"}),
        ("pollen-robotics/reachy-mini-emotions-library", "../x", {"dataset": "other"}),
    ],
)
def test_moves_are_named_only_for_pollen_libraries(
    client: tm.Telemetry, dataset: str, move: str, expected: dict[str, str]
) -> None:
    """Other datasets may be private: only Pollen's own libraries are named."""
    _start(client)
    client.note_move(dataset, move)
    props = _props(client, "move_played")[0]
    assert {k: props[k] for k in ("dataset", "move") if k in props} == expected


def test_client_hello_over_the_data_channel(
    client: tm.Telemetry, monkeypatch: pytest.MonkeyPatch, sim_backend: Any
) -> None:
    """The backend routes a WebRTC hello to telemetry, and peer loss ends it."""
    monkeypatch.setattr(tm, "telemetry", client)
    _start(client)
    hello = ClientHelloCmd(client="web/1.2.0", app="pollen-robotics/emotions")
    sim_backend.process_command(hello, send_response=lambda r: None, peer_id="p1")
    sim_backend.process_command(hello, send_response=lambda r: None)  # no peer
    sim_backend._on_peer_disconnect("p1")
    assert len(_props(client, "js_app_started")) == 1
    assert len(_props(client, "js_app_stopped")) == 1


@pytest.mark.parametrize(
    "query,headers,kind",
    [
        ("?client=desktop/0.9.5%20(macos)", {}, "desktop"),
        ("", {tm.CLIENT_HEADER: "desktop/0.9.5 (macos)"}, "desktop"),
        ("", {}, "move_listener"),
    ],
)
def test_move_updates_socket_counts_as_use(
    client: tm.Telemetry,
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    headers: dict[str, str],
    kind: str,
) -> None:
    """The desktop's long-lived socket is a usage session, named if it says so."""
    from reachy_mini.daemon.app.routers import move as move_router

    monkeypatch.setattr(move_router, "telemetry", client)
    _start(client)
    app = FastAPI()
    app.include_router(move_router.router, prefix="/api")
    with TestClient(app).websocket_connect(
        f"/api/move/ws/updates{query}", headers=headers
    ):
        assert [s[0] for s in client._sessions.values()] == [kind]
    assert not client._sessions
    # The header is reported by the middleware, not this route.
    connected = _props(client, "client_connected")
    assert [c["client_type"] for c in connected] == (["desktop"] if query else [])


def test_use_while_opted_out_is_not_reported_after_opting_in(
    client: tm.Telemetry,
) -> None:
    """Off, use, on, tick: only what is still running counts, nothing piles up."""
    _start(client)
    client.set_enabled(False)
    client.session_begin("app:x", "python_app", "pollen-robotics/hello")
    for i in range(300):
        client.session_begin(f"peer:{i}", "web", "pollen-robotics/emotions")
        client.session_end(f"peer:{i}")
    assert client._window == {}

    client.set_enabled(True)
    client._usage_tick()
    ticks = _props(client, "usage_tick")
    assert [t["kinds"] for t in ticks] == [["python_app"]]
    assert ticks[0]["apps"] == ["pollen-robotics/hello"]


def test_usage_tick_reports_the_seconds_in_use(client: tm.Telemetry) -> None:
    """active_s is the time anything was open, overlaps counted once."""
    now = [0.0]
    client._clock = lambda: now[0]
    _start(client)

    def at(t: float) -> None:
        now[0] = t

    at(10)
    client.session_begin("app:x", "python_app")
    at(20)
    client.session_begin("sdk:1", "python_sdk")  # the app's own SDK: overlaps
    at(30)
    client.session_end("app:x")
    at(40)
    client.session_end("sdk:1")
    at(150)
    client.session_begin("peer:p", "mobile")
    at(200)
    client._usage_tick()  # 10..40 and 150..200
    at(260)
    client._usage_tick()  # the phone stays connected for the whole window
    at(300)
    client._usage_tick()  # still connected
    client.peer_closed("p")
    at(400)
    client._usage_tick()  # it left exactly at the last tick: 0 s, no tick

    assert [t["active_s"] for t in _props(client, "usage_tick")] == [80, 60, 40]


def test_hello_queued_before_a_disconnect_leaves_no_session(
    client: tm.Telemetry, monkeypatch: pytest.MonkeyPatch, sim_backend: Any
) -> None:
    """The media server's disconnect (GLib thread) runs after a queued hello."""
    monkeypatch.setattr(tm, "telemetry", client)
    _start(client)
    loop = asyncio.new_event_loop()
    try:
        sim_backend._log_loop = loop
        hello = ClientHelloCmd(client="web/1.2.0", app="pollen-robotics/emotions")
        # As the media server does: messages go through the loop's queue...
        loop.call_soon_threadsafe(
            sim_backend._handle_webrtc_message, "p1", hello.model_dump_json()
        )
        # ...while the disconnect arrives first, from another thread.
        sim_backend._on_peer_disconnect("p1")
        loop.run_until_complete(asyncio.sleep(0))
    finally:
        sim_backend._log_loop = None
        loop.close()
    assert not client._sessions
    assert len(_props(client, "js_app_stopped")) == 1


def _run_ticks(
    client: tm.Telemetry, now: list[float], until: float, events: dict[float, Any]
) -> None:
    """Drive _maybe_tick like the sender thread does: once a flush interval."""
    next_tick = now[0] + tm.USAGE_TICK_S
    while now[0] < until:
        now[0] += tm.FLUSH_INTERVAL_S
        for t in [t for t in sorted(events) if t <= now[0]]:
            events.pop(t)()
        next_tick = client._maybe_tick(next_tick)


def test_a_short_run_ticks_once_when_use_stops(client: tm.Telemetry) -> None:
    """A 2-minute run is sent at the next flush, not lost waiting for the hour."""
    now = [0.0]
    client._clock = lambda: now[0]
    _start(client)
    events = {
        100.0: lambda: client.session_begin("app:x", "python_app"),
        220.0: lambda: client.session_end("app:x"),
    }
    # One flush after use stops: the daemon may well be gone by the hour.
    _run_ticks(client, now, 220.0 + tm.FLUSH_INTERVAL_S, events)
    assert [t["active_s"] for t in _props(client, "usage_tick")] == [120]


def test_a_long_run_ticks_hourly_then_when_it_stops(client: tm.Telemetry) -> None:
    """Hourly while use goes on, then the last partial hour when it stops."""
    now = [0.0]
    client._clock = lambda: now[0]
    _start(client)
    client.session_begin("app:x", "python_app")
    events = {9000.0: lambda: client.session_end("app:x")}  # 2.5 hours
    _run_ticks(client, now, 9000.0 + tm.FLUSH_INTERVAL_S, events)
    ticks = [t["active_s"] for t in _props(client, "usage_tick")]
    assert ticks == [3600, 3600, 1800]


def test_stop_sends_the_use_since_the_last_tick(
    client: tm.Telemetry, isolated: MagicMock
) -> None:
    """A 40-minute app then a clean stop: the final flush carries its tick."""
    now = [0.0]
    client._clock = lambda: now[0]
    _start(client)
    client.session_begin("app:x", "python_app", "pollen-robotics/hello")
    now[0] = 2400.0
    client.stop()
    sent = [e for call in isolated.call_args_list for e in call.kwargs["json"]["batch"]]
    ticks = [e["properties"] for e in sent if e["event"] == "usage_tick"]
    assert [t["active_s"] for t in ticks] == [2400]
