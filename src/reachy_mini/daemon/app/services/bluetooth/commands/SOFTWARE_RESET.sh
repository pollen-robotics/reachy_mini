#!/usr/bin/env bash

rm -rf /venvs/
cp -r /restore/venvs/ /
chown -R pollen:pollen /venvs

# Fix service file to point to /venvs/src (factory editable install location)
cat > /tmp/reachy-mini-daemon.service << 'EOF'
[Unit]
Description=Reachy Mini AP Launcher Service
After=network.target

[Service]
Type=simple
ExecStart=/venvs/src/reachy_mini/src/reachy_mini/daemon/app/services/wireless/launcher.sh
Restart=on-failure
User=pollen
WorkingDirectory=/venvs/src/reachy_mini/src/reachy_mini/daemon/app/services/wireless

[Install]
WantedBy=multi-user.target
EOF

cp /tmp/reachy-mini-daemon.service /etc/systemd/system/reachy-mini-daemon.service
systemctl daemon-reload
systemctl restart reachy-mini-daemon.service

