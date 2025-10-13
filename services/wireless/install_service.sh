#!/bin/bash

SERVICE_NAME="reachy-mini-daemon"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
LAUNCHER_PATH="$(pwd)/launcher.sh"

# Create the service file
cat <<EOF | sudo tee $SERVICE_FILE > /dev/null
[Unit]
Description=Reachy Mini AP Launcher Service
After=network.target

[Service]
Type=simple
ExecStart=$LAUNCHER_PATH
Restart=on-failure
User=$(whoami)
WorkingDirectory=$(dirname "$LAUNCHER_PATH")

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd, enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable --now $SERVICE_NAME

echo "Service '$SERVICE_NAME' installed and started."