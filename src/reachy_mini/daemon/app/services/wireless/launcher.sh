#!/bin/bash
source /venvs/mini_daemon/bin/activate
export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:/opt/gst-plugins-rs/lib/aarch64-linux-gnu/:/usr/local/lib/aarch64-linux-gnu/gstreamer-1.0/
export PATH=$PATH:/opt/uv
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/aarch64-linux-gnu/
export LIBCAMERA_IPA_MODULE_PATH=/usr/local/lib/aarch64-linux-gnu/libcamera/ipa
export LIBCAMERA_IPA_CONFIG_PATH=/usr/local/share/libcamera/ipa

# Ensure WiFi is not soft-blocked (can happen after a crash or kernel module reload)
sudo rfkill unblock wifi

# Run Python in unbuffered mode (-u) to ensure logs are immediately forwarded to systemd
python -u -m reachy_mini.daemon.app.main --wireless-version --no-wake-up-on-start
