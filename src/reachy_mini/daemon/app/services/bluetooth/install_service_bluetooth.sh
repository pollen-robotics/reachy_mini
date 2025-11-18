#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SERVICE_NAME="reachy-mini-bluetooth"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
LAUNCHER_PATH="$SCRIPT_DIR/bluetooth_service.py"
COMMANDS_DIR="$SCRIPT_DIR/commands"
SERVICE_PATH=/bluetooth/bluetooth_service.py

sudo cp "$LAUNCHER_PATH" "$SERVICE_PATH"
sudo cp -r "$COMMANDS_DIR" /bluetooth/commands
# Ensure Python script is executable
sudo chmod +x "$SERVICE_PATH"

# Create the systemd service file
cat <<EOF | sudo tee "$SERVICE_FILE" > /dev/null
[Unit]
Description=Reachy Mini Bluetooth GATT Service
After=network.target bluetooth.target
Requires=bluetooth.target

[Service]
Type=simple
ExecStart=/bin/bash -c 'sudo /usr/sbin/rfkill unblock bluetooth && sleep 2 && /usr/bin/python3 /bluetooth/bluetooth_service.py'
Restart=on-failure
User=$(whoami)
WorkingDirectory=$(dirname "$SERVICE_PATH")

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd, enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE_NAME"

echo "Service '$SERVICE_NAME' installed and started."
