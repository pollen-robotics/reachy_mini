#!/usr/bin/env bash

nmcli device disconnect wlan0
sleep 5
rfkill unblock wifi
systemctl restart reachy-mini-daemon.service 

