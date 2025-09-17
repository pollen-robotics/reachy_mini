import pytest

pytest_plugins = ('pytest_asyncio',)


def test_import():  # noqa: D100, D103
    from reachy_mini import ReachyMini  # noqa: F401
    from reachy_mini import ReachyMiniApp  # noqa: F401
    
@pytest.mark.asyncio
async def test_daemon():  # noqa: D100, D103
    from reachy_mini.daemon.daemon import Daemon    
    
    daemon = Daemon()
    await daemon.start(
        sim=True,
        headless=True,
        wake_up_on_start=False,
    )
    await daemon.stop(goto_sleep_on_stop=False)
