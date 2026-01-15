#!/usr/bin/env bash

rm -rf /venvs/
cp -r /restore/venvs/ /
chown -R pollen:pollen /venvs
systemctl restart reachy-mini-daemon.service

