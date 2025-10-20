import asyncio
import numpy as np
import pytest

from reachy_mini.daemon.daemon import Daemon    
from reachy_mini.reachy_mini import ReachyMini

@pytest.mark.asyncio
async def test_daemon_start_stop() -> None:
    from reachy_mini.daemon.daemon import Daemon    
    
    daemon = Daemon()
    await daemon.start(
        sim=True,
        headless=True,
        wake_up_on_start=False,
    )
    await daemon.stop(goto_sleep_on_stop=False)


@pytest.mark.asyncio
async def test_daemon_multiple_start_stop() -> None:    
    daemon = Daemon()
    
    for _ in range(3):
        await daemon.start(
            sim=True,
            headless=True,
            wake_up_on_start=False,
        )
        await daemon.stop(goto_sleep_on_stop=False)


@pytest.mark.asyncio
async def test_daemon_client_disconnection() -> None:
    daemon = Daemon()
    await daemon.start(
        sim=True,
        headless=True,
        wake_up_on_start=False,
    )

    client_connected = asyncio.Event()

    async def simple_client() -> None:
        with ReachyMini(media_backend="no_media") as mini:
            status = mini.client.get_status()
            assert status['state'] == "running"
            assert status['simulation_enabled']
            assert status['error'] is None
            assert status['backend_status']['motor_control_mode'] == "enabled"
            assert status['backend_status']['error'] is None
            assert status['wlan_ip'] is None
            client_connected.set()

    async def wait_for_client() -> None:
        await client_connected.wait()
        await daemon.stop(goto_sleep_on_stop=False)
    
    await asyncio.gather(simple_client(), wait_for_client())

@pytest.mark.asyncio
async def test_daemon_early_stop() -> None:
    daemon = Daemon()
    await daemon.start(
        sim=True,
        headless=True,
        wake_up_on_start=False,
    )

    client_connected = asyncio.Event()
    daemon_stopped = asyncio.Event()

    async def client_bg() -> None:
        with ReachyMini(media_backend="no_media") as reachy:
            client_connected.set()
            await daemon_stopped.wait()

            # Make sure the keep-alive check runs at least once
            reachy.client._check_alive_evt.clear()
            reachy.client._check_alive_evt.wait(timeout=100.0)

            with pytest.raises(ConnectionError, match="Lost connection with the server."):
                reachy.set_target(head=np.eye(4))

    async def will_stop_soon() -> None:
        await client_connected.wait()
        await daemon.stop(goto_sleep_on_stop=False)
        daemon_stopped.set()

    await asyncio.gather(client_bg(), will_stop_soon())

