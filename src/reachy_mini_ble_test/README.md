# Over-the-air BLE tests

Smoke tests for the robot's BLE provisioning service
(`reachy_mini/daemon/app/services/bluetooth/bluetooth_service.py`), run
**over the air from an external host** — not on the robot, and not in CI.
Typical setup: a Raspberry Pi (or any Linux box with BlueZ and a Bluetooth
adapter) in radio range of a powered Reachy Mini.

The suite discovers the robot by itself: it scans for the advertised name
(`ReachyMini`), connects, and exercises the GATT command protocol. Protocol
constants (UUIDs, etc.) are imported from the firmware source shipped in the
same package, so the suite fails loudly if the protocol drifts.

## Scope

Everything here is **read-only and safe**:

- discovery, connection, GATT layout, Device Information service
- status characteristics (network status, system status, available commands,
  hardware ID)
- synchronous command round-trips (`PING`, `ECHO`, journal streaming
  start/read/stop)
- public WiFi provisioning surface (`WIFI_STATUS` without the saved-network
  list, `WIFI_KEYEX`)
- auth gates: privileged commands (`WIFI_SCAN`, `CMD_*`) are refused without
  a PIN session

No PIN is ever sent (the robot-global wrong-PIN lockout is never armed), no
robot state is mutated, and no `CMD_*` script is executed. Tests for commands
the deployed firmware may predate skip with an explanation instead of failing.

## Running

On any Linux host with BlueZ (the suite needs nothing beyond `pytest` —
scanning shells out to `bluetoothctl`, the connection uses a raw LE socket):

```bash
pip install reachy-mini pytest
pytest --pyargs reachy_mini_ble_test -v
```

On a bare test host where the full dependency set won't build (e.g. a stock
Raspberry Pi without PyGObject's system headers), a minimal install is
enough — this package deliberately imports nothing from the rest of
`reachy_mini`:

```bash
pip install --no-deps reachy-mini
pip install pytest
pytest --pyargs reachy_mini_ble_test -v
```

All the usual pytest options apply (`-k ping`, `-m ble`, `--tb=short`, …).
From a repo checkout: `uv run pytest src/reachy_mini_ble_test -v`. The suite
is outside `testpaths`, so a plain `pytest` run never collects it.

## Configuration

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `REACHY_BLE_NAME` | `ReachyMini` | Advertised name to match (normalized, so `reachy-mini` also matches) |
| `REACHY_BLE_ADDRESS` | *(any)* | Pin discovery to one MAC address, e.g. when several robots are in range |
| `REACHY_BLE_SCAN_TIMEOUT` | `15` | BLE scan duration, seconds |
| `REACHY_BLE_RESPONSE_TIMEOUT` | `25` | Max wait for a proxied command's notification, seconds |

## Gotchas

- The robot accepts a **single BLE central**: if a phone/laptop is already
  connected (e.g. the web provisioning tool), discovery or connection fails.
- Daemon-proxied commands ack immediately with `OK: working` and deliver the
  real result as a notification on the response characteristic; the helper in
  `ble_link.py` handles both shapes.
- The suite connects through a raw LE ATT socket (`att_client.py`), not
  through BlueZ's `Device1.Connect`. Reason: the robot's controller is
  dual-mode and its advert lacks the "BR/EDR Not Supported" flag, so BlueZ
  centrals pick the classic BR/EDR bearer and fail with
  `br-connection-profile-unavailable`. Phones and Web Bluetooth are LE-only
  and never hit this; the raw socket behaves like them and tests the robot
  exactly as shipped.
- Future authenticated tests must respect the robot-global wrong-PIN
  throttle: a failed PIN arms a lockout (up to 300 s) that **survives
  disconnects** and would poison subsequent tests.
