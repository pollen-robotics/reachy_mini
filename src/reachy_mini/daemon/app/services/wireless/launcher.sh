#!/bin/bash
/venvs/src/reachy_mini/src/reachy_mini/daemon/app/services/wireless/generate_asoundrc.sh
source /venvs/mini_daemon/bin/activate
export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:/opt/gst-plugins-rs/lib/aarch64-linux-gnu/
# Run Python in unbuffered mode (-u) to ensure logs are immediately forwarded to systemd
python -u -m reachy_mini.daemon.app.main --wireless-version --no-autostart
