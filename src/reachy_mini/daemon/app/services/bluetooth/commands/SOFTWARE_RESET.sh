#!/usr/bin/env bash

rm -rf /venvs/
cp -a /restore/venvs/ /
chown -R pollen:pollen /venvs
systemctl restart reachy-mini-daemon.service

