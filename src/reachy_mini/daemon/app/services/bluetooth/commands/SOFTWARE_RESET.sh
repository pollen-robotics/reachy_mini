#!/usr/bin/env bash

rm -rf /venvs/mini_daemon/
cp -r /restore/mini_daemon /venvs/mini_daemon/
chown -R pollen:pollen /venvs/mini_daemon/
systemctl restart reachy-mini-daemon.service

