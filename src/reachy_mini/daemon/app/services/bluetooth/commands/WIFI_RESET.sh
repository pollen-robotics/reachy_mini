#!/usr/bin/env bash

nmcli --escape yes -t -f NAME,TYPE connection show | grep ':802-11-wireless$' | while IFS= read -r line; do
  conn="${line%:802-11-wireless}"
  conn="${conn//\\:/\:}"
  [ "$conn" != "Hotspot" ] && nmcli connection delete "$conn" 2>/dev/null
done
systemctl restart reachy-mini-daemon.service
