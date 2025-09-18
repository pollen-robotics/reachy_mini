import asyncio
import time
import numpy as np
import pytest

from reachy_mini.daemon.daemon import Daemon    
from reachy_mini.reachy_mini import ReachyMini

pytest_plugins = ('pytest_asyncio',)

@pytest.mark.asyncio
async def test_daemon_early_stop():    
    daemon = Daemon()
    await daemon.start(
        sim=True,
        headless=True,
        wake_up_on_start=False,
    )

    def client_bg():
        with pytest.raises(ConnectionError, match="Lost connection with the server."):
            with ReachyMini() as reachy:
                start = time.time()
                while time.time() - start < 5.0:
                    reachy.set_target(head=np.eye(4))
                    time.sleep(0.1)
                raise Exception("This should not happen")
    client_task = asyncio.get_event_loop().run_in_executor(None, client_bg)

    async def will_stop_soon():
        await asyncio.sleep(0.1)
        await daemon.stop(goto_sleep_on_stop=False)

    await asyncio.gather(client_task, will_stop_soon())

    

