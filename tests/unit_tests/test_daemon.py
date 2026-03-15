import asyncio
import threading

import numpy as np
import pytest
import uvicorn

from reachy_mini.daemon.app.main import Args, create_app
from reachy_mini.daemon.daemon import Daemon
from reachy_mini.reachy_mini import ReachyMini


async def _start_app_server(
    **daemon_kwargs: object,
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
        fastapi_port=0,  # let OS pick a free port
    )

    app = create_app(args)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait until the server is accepting connections
    while not server.started:
        await asyncio.sleep(0.05)

    sockets = server.servers[0].sockets
    port: int = sockets[0].getsockname()[1]

    return app.state.daemon, server, thread, port


async def _stop_app_server(server: uvicorn.Server, thread: threading.Thread) -> None:
    """Gracefully shut down the uvicorn server."""
    server.should_exit = True
    thread.join(timeout=10)


@pytest.mark.asyncio
async def test_daemon_start_stop() -> None:
    daemon, server, thread, _port = await _start_app_server()
    await daemon.stop(goto_sleep_on_stop=False)
    await _stop_app_server(server, thread)


@pytest.mark.asyncio
async def test_daemon_multiple_start_stop() -> None:
    daemon, server, thread, _port = await _start_app_server()
    await daemon.stop(goto_sleep_on_stop=False)
    await _stop_app_server(server, thread)

    # Start a second time with a fresh server
    daemon2, server2, thread2, _port2 = await _start_app_server()
    await daemon2.stop(goto_sleep_on_stop=False)
    await _stop_app_server(server2, thread2)


@pytest.mark.asyncio
async def test_daemon_client_disconnection() -> None:
    daemon, server, thread, port = await _start_app_server()

    client_connected = asyncio.Event()

    async def simple_client() -> None:
        with ReachyMini(host="localhost", port=port, media_backend="no_media") as mini:
            status = mini.client.get_status()
            assert status.state == "running"
            assert status.simulation_enabled
            assert status.error is None
            assert status.backend_status is not None
            assert status.backend_status.motor_control_mode == "enabled"
            assert status.backend_status.error is None
            assert status.wlan_ip is None
            client_connected.set()

    async def wait_for_client() -> None:
        await client_connected.wait()
        await daemon.stop(goto_sleep_on_stop=False)
        await _stop_app_server(server, thread)

    await asyncio.gather(simple_client(), wait_for_client())


@pytest.mark.asyncio
async def test_daemon_early_stop() -> None:
    daemon, server, thread, port = await _start_app_server()

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
        await _stop_app_server(server, thread)
        daemon_stopped.set()

    await asyncio.gather(client_bg(), will_stop_soon())


@pytest.mark.asyncio
async def test_multi_robot_isolation() -> None:
    """Two daemons on different ports must be fully independent.

    Commands sent to one robot must not affect the other.
    """
    daemon1, server1, thread1, port1 = await _start_app_server()
    daemon2, server2, thread2, port2 = await _start_app_server()

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
        await daemon2.stop(goto_sleep_on_stop=False)
        await _stop_app_server(server1, thread1)
        await _stop_app_server(server2, thread2)
