#!/bin/bash
source /venvs/mini_daemon/bin/activate
export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:/opt/gst-plugins-rs/lib/aarch64-linux-gnu/:/usr/local/lib/aarch64-linux-gnu/gstreamer-1.0/
export PATH=$PATH:/opt/uv
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/aarch64-linux-gnu/
export LIBCAMERA_IPA_MODULE_PATH=/usr/local/lib/aarch64-linux-gnu/libcamera/ipa
export LIBCAMERA_IPA_CONFIG_PATH=/usr/local/share/libcamera/ipa

# Cap glibc malloc arenas. On a quad core Pi 4 the default (8 x ncpu) lets the
# multithreaded daemon spread allocations across many 64MB arenas, which grows
# RSS over time on the 4GB Wireless unit. 2 keeps it bounded. See issue #1165.
export MALLOC_ARENA_MAX=2

# Ensure WiFi is not soft-blocked (can happen after a crash or kernel module reload)
sudo rfkill unblock wifi

# Pre-warm the apps venv while the daemon boots. On a cold boot the first app
# start spends most of its time reading .pyc/.so files from the SD card; this
# background import pulls them into the page cache in parallel with daemon
# init. Idle CPU/IO priority keeps it from competing with the daemon, and any
# failure is harmless (the first app start is then simply as slow as today).
if [ -x /venvs/apps_venv/bin/python ]; then
    nice -n 19 ionice -c 3 /venvs/apps_venv/bin/python -c \
        "import reachy_mini_conversation_app.main" >/dev/null 2>&1 &
fi

# Run Python in unbuffered mode (-u) to ensure logs are immediately forwarded to systemd
python -u -m reachy_mini.daemon.app.main --wireless-version --no-wake-up-on-start
