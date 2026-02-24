"""Pre-warmed Python launcher for Reachy Mini apps.

Pre-imports common heavy packages used by Reachy Mini apps, then waits
for a module name on stdin. When received, runs the module using runpy
(equivalent to ``python -m <module>``).

On the CM4, importing numpy + zenoh + fastapi + pydantic + reachy_mini
takes ~3.5 s.  By doing this work *before* the user clicks "Start",
those imports are already cached in ``sys.modules`` and the app starts
almost instantly.

Protocol
--------
1. The daemon spawns this script in the apps-venv Python.
2. This script imports common packages, then prints
   ``WARM_READY <elapsed_seconds>`` to stdout.
3. The daemon reads that line and keeps the process idle.
4. When ``start_app()`` is called, the daemon writes
   ``<module_name>\n`` to this process's stdin.
5. This script runs the module via ``runpy.run_module``,
   exactly as ``python -m <module_name>`` would.
"""

import sys
import time as _time

_t0 = _time.perf_counter()

# Pre-import heavy packages (these take ~3.5 s on CM4).
import numpy  # noqa: F401
import zenoh  # noqa: F401
import pydantic  # noqa: F401
import fastapi  # noqa: F401
import reachy_mini  # noqa: F401

# Also pre-import commonly used sub-modules so apps that reference
# them (like marionette) get instant imports.
import reachy_mini.motion.recorded_move  # noqa: F401
import reachy_mini.utils.interpolation  # noqa: F401

_elapsed = _time.perf_counter() - _t0

# Signal ready to parent process.
print(f"WARM_READY {_elapsed:.3f}", flush=True)

# Block until the parent sends a module name.
module_name = sys.stdin.readline().strip()
if not module_name:
    sys.exit(0)

# Run the module exactly as ``python -m <module_name>`` would.
import runpy

runpy.run_module(module_name, run_name="__main__", alter_sys=True)
