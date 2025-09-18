import asyncio
import numpy as np
import pytest

from reachy_mini.daemon.daemon import Daemon    
from reachy_mini.reachy_mini import ReachyMini


@pytest.mark.asyncio
async def test_daemon_early_stop():    
    daemon = Daemon()
    await daemon.start(
        sim=True,
        headless=True,
        wake_up_on_start=False,
    )

    client_connected = asyncio.Event()
    daemon_stopped = asyncio.Event()

    async def client_bg():
        with ReachyMini() as reachy:
            client_connected.set()
            await daemon_stopped.wait()

            # Make sure the keep-alive check runs at least once
            await asyncio.sleep(1.1)

            with pytest.raises(ConnectionError, match="Lost connection with the server."):
                reachy.set_target(head=np.eye(4))
        
    async def will_stop_soon():
        await client_connected.wait()
        await daemon.stop(goto_sleep_on_stop=False)
        daemon_stopped.set()

    await asyncio.gather(client_bg(), will_stop_soon())


