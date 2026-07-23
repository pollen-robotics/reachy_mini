"""The daemon must not pay for onnxruntime at boot.

onnxruntime costs ~1-1.5s of CPU on the wireless robot's CM4. It is only
needed once head tracking is actually enabled (FaceTracker) or when the NN
kinematics engine is explicitly selected — neither happens during boot, where
the default engine is AnalyticalKinematics. Two eager routes used to exist:

- ``daemon.backend.abstract`` imported ``vision.face_tracking`` at module level;
- ``kinematics/__init__`` imported ``nn_kinematics`` at module level, paid by
  ``Backend.__init__`` when it fetches ``AnalyticalKinematics``.

This test runs in a subprocess so imports already performed by other tests
cannot mask a regression.
"""

import subprocess
import sys

_SCRIPT = """
import sys

import reachy_mini.daemon.backend.abstract  # noqa: F401

for banned in ("onnxruntime", "reachy_mini.vision.face_tracking"):
    assert banned not in sys.modules, f"importing the daemon backend pulled {banned}"

from reachy_mini.daemon.backend.mockup_sim.backend import MockupSimBackend

MockupSimBackend(use_audio=False)  # default engine: AnalyticalKinematics
assert "onnxruntime" not in sys.modules, "Backend.__init__ pulled onnxruntime"

print("LAZY_OK")
sys.stdout.flush()
import os

os._exit(0)  # don't wait on any backend helper threads
"""


def test_daemon_backend_does_not_import_onnxruntime() -> None:
    """Importing and constructing a backend must leave onnxruntime unloaded."""
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"subprocess exited with {result.returncode}.\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
    assert "LAZY_OK" in result.stdout
