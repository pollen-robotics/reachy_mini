#!/bin/bash
source /home/pollen/venvs/mini_daemon/bin/activate
python -m reachy_mini.daemon.app.main --wireless-version --no-autostart
