#!/bin/bash
/venvs/src/reachy_mini/src/reachy_mini/daemon/app/services/wireless/generate_asoundrc.sh
source /venvs/mini_daemon/bin/activate
export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:/opt/gst-plugins-rs/lib/aarch64-linux-gnu/
python -m reachy_mini.daemon.app.main --wireless-version --stream --no-autostart
