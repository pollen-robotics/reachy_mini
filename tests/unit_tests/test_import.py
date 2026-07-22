"""Import isolation tests."""

import subprocess
import sys
import textwrap


def test_import_defers_daemon_backend() -> None:
    """The public SDK import excludes the daemon backend module."""
    script = textwrap.dedent(
        """
        import sys

        import reachy_mini

        assert "reachy_mini.daemon.backend.abstract" not in sys.modules

        from reachy_mini import ReachyMini, ReachyMiniApp
        from reachy_mini.io import WSServer
        from reachy_mini.io.ws_server import WSServer as DirectWSServer

        assert WSServer is DirectWSServer
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True)
