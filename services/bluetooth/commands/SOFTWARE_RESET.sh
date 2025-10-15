#!/usr/bin/env bash

rm -rf /home/pollen/venvs/mini_daemon/
mkdir -p /home/pollen/venvs
cp -r /restore/mini_daemon /home/pollen/venvs/mini_daemon/
chown -R pollen:pollen /home/pollen/venvs/mini_daemon/
systemctl restart reachy-mini-daemon.service

