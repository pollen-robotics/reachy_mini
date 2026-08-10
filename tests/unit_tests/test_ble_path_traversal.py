"""The BLE ``CMD_`` handler must never sudo-execute a path outside ``commands/``."""
import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="BLE provisioning service is Linux-only"
)

_SERVICE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/reachy_mini/daemon/app/services/bluetooth/bluetooth_service.py"
)


def _import_bluetooth_service():
    """Load bluetooth_service with a stubbed ``dbus`` (absent from the venv)."""
    service = types.ModuleType("dbus.service")

    class _Object:
        def __init__(self, *a, **k):
            pass

    def _decorator_factory(*a, **k):
        return lambda fn: fn

    service.Object = _Object
    service.method = _decorator_factory
    service.signal = _decorator_factory

    exceptions = types.ModuleType("dbus.exceptions")

    class DBusException(Exception):
        pass

    exceptions.DBusException = DBusException

    mainloop = types.ModuleType("dbus.mainloop")
    mainloop_glib = types.ModuleType("dbus.mainloop.glib")
    mainloop_glib.DBusGMainLoop = MagicMock()
    mainloop.glib = mainloop_glib

    dbus = types.ModuleType("dbus")
    dbus.service = service
    dbus.exceptions = exceptions
    dbus.mainloop = mainloop
    dbus.__getattr__ = lambda name: MagicMock()  # any other dbus.X at import

    stubs = {
        "dbus": dbus,
        "dbus.service": service,
        "dbus.exceptions": exceptions,
        "dbus.mainloop": mainloop,
        "dbus.mainloop.glib": mainloop_glib,
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "_ble_path_traversal_service_under_test", _SERVICE_PATH
        )
        bt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bt)
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return bt


# Imported once at module scope; only runs on Linux thanks to the guard.
if sys.platform == "linux":
    bt = _import_bluetooth_service()


@pytest.fixture
def authed_service(tmp_path, monkeypatch):
    """Service whose CWD has a ``commands/`` dir and that is already paired."""
    (tmp_path / "commands").mkdir()
    monkeypatch.chdir(tmp_path)

    svc = bt.BluetoothCommandService(device_name="ReachyMini", pin_code="00000")
    # The CMD_ branch requires a prior successful PIN_ authentication.
    svc.connected = True
    return svc


def _assert_no_traversal(mock_run, commands_dir: Path, payload: bytes) -> None:
    """Fail iff ``mock_run`` ran ``sudo`` on a path outside ``commands_dir``."""
    if not mock_run.called:
        return  # Handler refused — safe.

    cmd_args = mock_run.call_args.args[0]
    invoked_path = Path(cmd_args[1]).resolve()
    assert invoked_path.is_relative_to(commands_dir), (
        f"PATH TRAVERSAL: handler invoked `sudo {invoked_path}` for payload "
        f"{payload!r}, which escapes the intended commands directory "
        f"{commands_dir}. The handler must reject `..` segments and absolute "
        f"paths, or canonicalize and confirm containment."
    )


def test_handler_rejects_relative_path_traversal(authed_service, tmp_path):
    """``CMD_../attacker/pwn`` must not be sudo-executed outside ``commands/``."""
    evil_dir = tmp_path / "attacker"
    evil_dir.mkdir()
    evil_script = evil_dir / "pwn.sh"
    evil_script.write_text("#!/bin/sh\necho pwned\n")
    evil_script.chmod(0o755)

    # CMD_ + "../attacker/pwn" -> os.path.join("commands", "../attacker/pwn.sh")
    payload = b"CMD_../attacker/pwn"

    with patch.object(bt.subprocess, "run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        authed_service._handle_command(payload)

    _assert_no_traversal(mock_run, (tmp_path / "commands").resolve(), payload)


def test_handler_rejects_absolute_path_payload(authed_service, tmp_path):
    """An absolute payload must not bypass ``commands/`` (join drops the prefix)."""
    target_dir = tmp_path / "elsewhere"
    target_dir.mkdir()
    target = target_dir / "rooted.sh"
    target.write_text("#!/bin/sh\necho rooted\n")
    target.chmod(0o755)

    # The trailing ".sh" is appended by the handler, so strip it from the payload.
    abs_target_no_ext = str(target).removesuffix(".sh")
    payload = f"CMD_{abs_target_no_ext}".encode()

    with patch.object(bt.subprocess, "run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        authed_service._handle_command(payload)

    _assert_no_traversal(mock_run, (tmp_path / "commands").resolve(), payload)
