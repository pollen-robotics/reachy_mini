"""Import isolation tests."""

import subprocess
import sys
import textwrap


def test_import_defers_daemon_server() -> None:
    """The public SDK import excludes daemon-only modules until requested."""
    script = textwrap.dedent(
        """
        import sys

        import reachy_mini

        assert "reachy_mini.io.ws_server" not in sys.modules

        from reachy_mini import ReachyMini, ReachyMiniApp
        from reachy_mini.io import WSServer
        from reachy_mini.io.ws_server import WSServer as DirectWSServer

        assert WSServer is DirectWSServer
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True)
