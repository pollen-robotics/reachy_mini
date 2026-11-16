#!/bin/bash
source /venvs/mini_daemon/bin/activate

# Run the monitor file directly instead of `python -m reachy_mini...`:
# module execution imports the whole reachy_mini SDK (several seconds of CPU
# and I/O during boot, tens of MB of RSS) while the script itself only needs
# gpiozero and the stdlib.
exec python "$(dirname "$(readlink -f "$0")")/shutdown_monitor.py"
