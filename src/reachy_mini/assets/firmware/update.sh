#!/bin/bash
# Firmware update script for Reachy Mini
# Usage: ./update.sh <firmware_file>
firmware="$1"
if [ -z "$firmware" ]; then
    echo "Usage: $0 <firmware_file>"
    exit 1
fi
dfu-util -R -e -a 1 -D "$firmware"