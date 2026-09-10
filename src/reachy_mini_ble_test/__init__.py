"""Over-the-air BLE smoke tests for the Reachy Mini provisioning service.

Run from any external Linux host with BlueZ and a Bluetooth adapter in radio
range of a powered robot — NOT on the robot itself:

    pip install reachy-mini pytest        # or: pip install --no-deps reachy-mini pytest
    pytest --pyargs reachy_mini_ble_test -v

This package deliberately depends on nothing but the standard library (plus
pytest to run), so a minimal ``--no-deps`` install is enough on a bare test
host like a Raspberry Pi. See README.md in this directory for scope,
configuration, and gotchas.
"""
