import asyncio
import re
import threading
import time
from typing import Generator

import numpy as np
import pytest
import uvicorn

from reachy_mini.daemon.app.main import CORS_ORIGIN_REGEX, Args, create_app
from reachy_mini.daemon.daemon import Daemon
from reachy_mini.reachy_mini import ReachyMini
from reachy_mini.utils.discovery import DiscoveredRobot

pytestmark = pytest.mark.usefixtures("no_daemon_mdns_advertising")

SHARED_ROBOT_NAME = "shared_robot"


def _start_app_server(
    robot_name: str = "reachy_mini",
) -> tuple[Daemon, uvicorn.Server, threading.Thread, int]:
    """Start a full FastAPI + daemon server in a background thread.

    Returns (daemon, server, thread, port).
    """
    args = Args(
        sim=True,
        headless=True,
        wake_up_on_start=False,
        no_media=True,
        autostart=True,
        robot_name=robot_name,
        fastapi_port=0,  # let OS pick a free port
    )

    app = create_app(args)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait until the server is accepting connections
    while not server.started:
        time.sleep(0.05)

    sockets = server.servers[0].sockets
    port: int = sockets[0].getsockname()[1]

    return app.state.daemon, server, thread, port


def _stop_app_server(server: uvicorn.Server, thread: threading.Thread) -> None:
    """Gracefully shut down the uvicorn server."""
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(scope="module")
def shared_daemon() -> Generator[tuple[Daemon, int], None, None]:
    """One daemon for every test that only connects to it, instead of one each.

    Tests using it must leave it running and as they found it. Those that stop
    the daemon or test its start/stop start their own.
    """
    daemon, server, thread, port = _start_app_server(robot_name=SHARED_ROBOT_NAME)
    try:
        yield daemon, port
    finally:
        asyncio.run(daemon.stop(goto_sleep_on_stop=False))
        _stop_app_server(server, thread)


@pytest.mark.asyncio
async def test_daemon_multiple_start_stop() -> None:
    daemon, server, thread, _port = _start_app_server()
    await daemon.stop(goto_sleep_on_stop=False)
    _stop_app_server(server, thread)

    # Start a second time with a fresh server
    daemon2, server2, thread2, _port2 = _start_app_server()
    await daemon2.stop(goto_sleep_on_stop=False)
    _stop_app_server(server2, thread2)


def test_daemon_client_disconnection(shared_daemon: tuple[Daemon, int]) -> None:
    daemon, port = shared_daemon

    with ReachyMini(host="localhost", port=port, media_backend="no_media") as mini:
        status = mini.client.get_status()
        assert status.state == "running"
        assert status.simulation_enabled
        assert status.error is None
        assert status.backend_status is not None
        assert status.backend_status.motor_control_mode == "enabled"
        assert status.backend_status.error is None
        assert status.wlan_ip is None

    # The client leaving doesn't take the daemon down with it.
    assert daemon.status().state == "running"


def test_sdk_warns_on_daemon_version_mismatch(
    shared_daemon: tuple[Daemon, int],
) -> None:
    daemon, port = shared_daemon
    real_version = daemon._status.version
    daemon._status.version = "0.0.0"

    try:
        with pytest.warns(
            RuntimeWarning,
            match="Reachy Mini SDK and daemon versions do not match",
        ):
            with ReachyMini(host="localhost", port=port, media_backend="no_media"):
                pass
    finally:
        daemon._status.version = real_version


def test_named_local_daemon_is_validated(shared_daemon: tuple[Daemon, int]) -> None:
    _daemon, port = shared_daemon

    with ReachyMini(
        robot_name=SHARED_ROBOT_NAME,
        port=port,
        media_backend="no_media",
    ) as mini:
        assert mini.connection_mode == "localhost_only"
        assert mini.client.host == "127.0.0.1"
        assert mini.robot_name == SHARED_ROBOT_NAME

    with pytest.raises(
        ConnectionError,
        match="local Reachy Mini daemon named 'other_robot'",
    ):
        ReachyMini(
            robot_name="other_robot",
            port=port,
            connection_mode="localhost_only",
            media_backend="no_media",
        )


def test_named_daemon_auto_discovers_and_tries_all_addresses(
    shared_daemon: tuple[Daemon, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _daemon, port = shared_daemon
    discovered = DiscoveredRobot(
        name=SHARED_ROBOT_NAME,
        host="",
        port=port,
        addresses=["127.0.0.2", "127.0.0.1"],
    )
    monkeypatch.setattr(
        "reachy_mini.reachy_mini.find_robots",
        lambda timeout: [discovered],
    )

    with ReachyMini(
        robot_name=SHARED_ROBOT_NAME,
        port=1,
        media_backend="no_media",
    ) as mini:
        assert mini.connection_mode == "network"
        assert mini.client.host == "127.0.0.1"
        assert mini.client.port == port


def test_named_daemon_falls_back_to_host_when_mdns_misses(
    shared_daemon: tuple[Daemon, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _daemon, port = shared_daemon
    monkeypatch.setattr(
        "reachy_mini.reachy_mini.find_robots",
        lambda timeout: [],
    )

    with ReachyMini(
        robot_name=SHARED_ROBOT_NAME,
        host="127.0.0.1",
        port=port,
        connection_mode="network",
        media_backend="no_media",
    ) as mini:
        assert mini.connection_mode == "network"
        assert mini.client.host == "127.0.0.1"
        assert mini.client.port == port


def test_named_daemon_host_fallback_rejects_wrong_name(
    shared_daemon: tuple[Daemon, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _daemon, port = shared_daemon
    monkeypatch.setattr(
        "reachy_mini.reachy_mini.find_robots",
        lambda timeout: [],
    )

    with pytest.raises(ConnectionError, match="via mDNS or host"):
        ReachyMini(
            robot_name="wanted_robot",
            host="127.0.0.1",
            port=port,
            connection_mode="network",
            media_backend="no_media",
        )


@pytest.mark.asyncio
async def test_daemon_early_stop() -> None:
    daemon, server, thread, port = _start_app_server()

    client_connected = asyncio.Event()
    daemon_stopped = asyncio.Event()

    async def client_bg() -> None:
        with ReachyMini(
            host="localhost", port=port, media_backend="no_media"
        ) as reachy:
            client_connected.set()
            await daemon_stopped.wait()

            # Make sure the keep-alive check runs at least once
            reachy.client._check_alive_evt.clear()
            reachy.client._check_alive_evt.wait(timeout=100.0)

            with pytest.raises(
                ConnectionError, match="Lost connection with the server."
            ):
                reachy.set_target(head=np.eye(4))

    async def will_stop_soon() -> None:
        await client_connected.wait()
        await daemon.stop(goto_sleep_on_stop=False)
        _stop_app_server(server, thread)
        daemon_stopped.set()

    await asyncio.gather(client_bg(), will_stop_soon())


@pytest.mark.asyncio
async def test_multi_robot_isolation(shared_daemon: tuple[Daemon, int]) -> None:
    """Two daemons on different ports must be fully independent.

    Commands sent to one robot must not affect the other. The shared daemon is
    the robot that must stay still, so it is left as the other tests expect.
    """
    daemon1, server1, thread1, port1 = _start_app_server()
    _daemon2, port2 = shared_daemon

    try:
        with (
            ReachyMini(host="localhost", port=port1, media_backend="no_media") as mini1,
            ReachyMini(host="localhost", port=port2, media_backend="no_media") as mini2,
        ):
            # Both robots should be running independently
            status1 = mini1.client.get_status()
            status2 = mini2.client.get_status()
            assert status1.state == "running"
            assert status2.state == "running"

            # Read initial antenna positions from both robots
            _, ant1_before = mini1.client.get_current_joints()
            _, ant2_before = mini2.client.get_current_joints()

            # Send antenna command ONLY to robot 1
            new_antennas = [0.5, -0.5]
            mini1.set_target_antenna_joint_positions(new_antennas)

            # Wait for the command to take effect
            await asyncio.sleep(0.5)

            _, ant1_after = mini1.client.get_current_joints()
            _, ant2_after = mini2.client.get_current_joints()

            # Robot 1 antennas should have moved toward the target
            delta1 = np.max(np.abs(np.array(ant1_after) - np.array(ant1_before)))
            assert delta1 > 0.1, (
                f"Robot 1 antennas did not move after command (max delta={delta1})"
            )

            # Robot 2 antennas should be untouched (only sim noise)
            delta2 = np.max(np.abs(np.array(ant2_after) - np.array(ant2_before)))
            assert delta2 < 0.01, (
                f"Robot 2 antennas moved after commanding robot 1 (max delta={delta2})"
            )

    finally:
        await daemon1.stop(goto_sleep_on_stop=False)
        _stop_app_server(server1, thread1)


def test_cors_origin_regex():
    """Starlette fullmatches Origin against CORS_ORIGIN_REGEX (GHSA-p4cp-8gwf-3fgv)."""
    pat = re.compile(CORS_ORIGIN_REGEX)
    allowed = [
        "http://localhost:3000",
        "https://localhost",
        "http://127.0.0.1:8000",
        "tauri://localhost",
        "https://tauri.localhost",
        "capacitor://localhost",
    ]
    blocked = [
        "http://evil.com",
        "http://localhost.evil.com",
        "https://reachy.tauri.localhost.evil.com",
        "tauri://evil",
        "capacitor://evil",
    ]
    for origin in allowed:
        assert pat.fullmatch(origin), origin
    for origin in blocked:
        assert not pat.fullmatch(origin), origin
