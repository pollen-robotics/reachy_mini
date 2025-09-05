def test_import():  # noqa: D100, D103
    from reachy_mini import ReachyMini  # noqa: F401
    from reachy_mini.app import ReachyMiniApp  # noqa: F401
    
def test_daemon():  # noqa: D100, D103
    from reachy_mini.daemon.daemon import Daemon    
    
    daemon = Daemon()
