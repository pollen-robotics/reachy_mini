#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SERVICE_NAME="gpio-shutdown-daemon"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
LAUNCHER_PATH="$SCRIPT_DIR/launcher.sh"

# Create the service file
cat <<EOF | sudo tee $SERVICE_FILE > /dev/null
[Unit]
Description=Reachy Mini GPIO Shutdown Daemon
After=multi-user.target

[Service]
Type=simple
ExecStart=$LAUNCHER_PATH
Restart=on-failure
User=$(whoami)
WorkingDirectory=$(dirname "$LAUNCHER_PATH")

[Install]
WantedBy=multi-user.target
EOF

chmod +x $LAUNCHER_PATH

# Reload systemd, enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable --now $SERVICE_NAME

echo "Service '$SERVICE_NAME' installed and started."