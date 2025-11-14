#!/bin/bash

set -e

UDEV_RULE="99-reachy-mini-audio.rules"
SRC_DIR="$(dirname "$0")"
RULE_PATH="$SRC_DIR/$UDEV_RULE"
DEST_PATH="/etc/udev/rules.d/$UDEV_RULE"

if [[ ! -f "$RULE_PATH" ]]; then
    echo "Error: $RULE_PATH not found."
    exit 1
fi

echo "Installing $UDEV_RULE to /etc/udev/rules.d/..."
sudo cp "$RULE_PATH" "$DEST_PATH"
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "udev rule installed successfully."