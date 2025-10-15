#!/usr/bin/env bash

nmcli device disconnect wlan0
systemctl restart reachy-mini-daemon.service 

