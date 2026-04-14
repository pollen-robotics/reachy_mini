"""Regression test for pollen-robotics/reachy-mini-desktop-app#185.

The `gstreamer_python` wheel on macOS ships a `libgstpython.dylib` that was built
against python.org's framework Python (`/Library/Frameworks/Python.framework/...`).
When the desktop app bootstraps its venv using uv's python-build-standalone
distribution, loading libgstpython into the process pulls a second CPython runtime
into the address space and the two collide during plugin scanning — surfacing as
`ModuleNotFoundError: No module named 'math'` and then a segfault.

This test spawns a subprocess that imports GStreamer and calls `Gst.init([])` —
the exact code path that crashes. It runs on both macOS and Linux because the
ABI mismatch is mechanically possible on any platform where libgstpython's linked
Python differs from the host Python.

CI previously missed this class of bug because the default pytest invocation is
`-m 'not audio and not video and not wireless'` — `Gst.init()` was never actually
exercised under standalone Python on macos-latest. This test has no such markers
so it always runs.
"""

import subprocess
import sys


def test_gst_init_does_not_crash_host_python() -> None:
    """`Gst.init([])` must complete cleanly under the same Python that hosts the tests.

    If libgstpython.dylib is linked against a different Python ABI than the current
    interpreter, this subprocess will segfault (exit code -11 / 139) or fail to
    import `math` during `.pth` processing — both conditions are caught here.
    """
    script = (
        "import sys\n"
        "try:\n"
        "    import gi\n"
        "    gi.require_version('Gst', '1.0')\n"
        "    from gi.repository import Gst\n"
        "except ImportError as e:\n"
        # PyGObject / gstreamer-bundle may legitimately not be installed in very
        # minimal test environments. Skip rather than fail in that case — the
        # regression we care about is the *crash*, not a missing optional dep.
        "    print(f'SKIP: {e}')\n"
        "    sys.exit(77)\n"
        "Gst.init([])\n"
        "version = Gst.version()\n"
        "print(f'GST_OK {version}')\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode == 77:
        import pytest

        pytest.skip(f"GStreamer Python bindings not installed: {result.stdout.strip()}")

    assert result.returncode == 0, (
        f"Gst.init subprocess exited with {result.returncode}.\n"
        f"stdout: {result.stdout!r}\n"
        f"stderr: {result.stderr!r}\n"
        "This is the pollen-robotics/reachy-mini-desktop-app#185 regression: "
        "libgstpython.dylib is linked against a Python that doesn't match the host "
        "interpreter's ABI. Check `otool -L` on "
        "<venv>/lib/python3.12/site-packages/gstreamer_python/lib/gstreamer-1.0/libgstpython.dylib."
    )
    assert "GST_OK" in result.stdout, (
        f"Expected 'GST_OK' marker in stdout, got: {result.stdout!r}"
    )


def test_gst_device_monitor_does_not_crash_host_python() -> None:
    """`Gst.DeviceMonitor` must not segfault while probing devices.

    The daemon uses ``Gst.DeviceMonitor`` during startup to find camera and audio
    hardware. On macOS this can crash inside the external registry scanner unless
    ``GST_REGISTRY_FORK=no`` is applied before ``Gst.init([])``.
    """
    script = (
        "import sys\n"
        "try:\n"
        "    import gi\n"
        "    gi.require_version('Gst', '1.0')\n"
        "    from gi.repository import Gst\n"
        "    from reachy_mini.media.gstreamer_utils import init_gst\n"
        "except ImportError as e:\n"
        "    print(f'SKIP: {e}')\n"
        "    sys.exit(77)\n"
        "init_gst()\n"
        "monitor = Gst.DeviceMonitor()\n"
        "monitor.add_filter('Video/Source')\n"
        "monitor.start()\n"
        "devices = monitor.get_devices()\n"
        "print(f'DEV_MON_OK {len(devices)}')\n"
        "monitor.stop()\n"
    )

    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode == 77:
        import pytest

        pytest.skip(f"GStreamer Python bindings not installed: {result.stdout.strip()}")

    assert result.returncode == 0, (
        f"Gst.DeviceMonitor subprocess exited with {result.returncode}.\n"
        f"stdout: {result.stdout!r}\n"
        f"stderr: {result.stderr!r}\n"
        "The daemon's media device probing is not safe in this Python/GStreamer "
        "runtime."
    )
    assert "DEV_MON_OK" in result.stdout, (
        f"Expected 'DEV_MON_OK' marker in stdout, got: {result.stdout!r}"
    )


def test_daemon_create_app_with_media_does_not_crash_host_python() -> None:
    """Creating the daemon app with media enabled must stay in-process alive.

    This exercises the exact media-initialisation path used by
    ``reachy-mini-daemon`` without opening the serial backend.
    """
    script = (
        "import sys\n"
        "try:\n"
        "    import reachy_mini.daemon.app.main as main\n"
        "except ImportError as e:\n"
        "    print(f'SKIP: {e}')\n"
        "    sys.exit(77)\n"
        "app = main.create_app(main.Args(fastapi_port=0, autostart=False, wake_up_on_start=False))\n"
        "daemon = app.state.daemon\n"
        "if daemon._media_server is not None:\n"
        "    daemon._media_server.stop()\n"
        "    daemon._media_server.close()\n"
        "    daemon._media_server = None\n"
        "print('CREATE_APP_OK')\n"
    )

    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode == 77:
        import pytest

        pytest.skip(f"GStreamer Python bindings not installed: {result.stdout.strip()}")

    assert result.returncode == 0, (
        f"create_app subprocess exited with {result.returncode}.\n"
        f"stdout: {result.stdout!r}\n"
        f"stderr: {result.stderr!r}\n"
        "The daemon's media initialisation path is not safe in this runtime."
    )
    assert "CREATE_APP_OK" in result.stdout, (
        f"Expected 'CREATE_APP_OK' marker in stdout, got: {result.stdout!r}"
    )
