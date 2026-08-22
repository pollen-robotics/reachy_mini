"""Unit tests for the IMU exposure on the REST state routes + state snapshot.

Two contracts are covered:

- **Fail-silent**: on hardware without an IMU (Lite, simulation) every
  surface answers ``null`` — HTTP 200, never an error.
- **No discriminator leak**: the backend returns an ``ImuDataMsg`` (the
  legacy WS message, carrying ``type="imu_data"``), but the REST and
  snapshot fields are typed ``ImuData``, so the ``type`` key must never
  reach a client.

The IMU device itself is never touched: ``get_imu_data`` is stubbed, which
is exactly the seam the real backend fills from its control-loop cache.
"""

from __future__ import annotations

import json

import pytest

from reachy_mini.daemon.app.routers import state
from reachy_mini.daemon.backend.mockup_sim.backend import MockupSimBackend
from reachy_mini.io.protocol import ImuDataMsg

READING = ImuDataMsg(
    accelerometer=[0.1, -0.2, 9.79],
    gyroscope=[0.01, 0.02, -0.03],
    quaternion=[1.0, 0.0, 0.0, 0.0],  # w-first
    temperature=27.5,
)

EXPECTED = {
    "accelerometer": [0.1, -0.2, 9.79],
    "gyroscope": [0.01, 0.02, -0.03],
    "quaternion": [1.0, 0.0, 0.0, 0.0],
    "temperature": 27.5,
}


@pytest.fixture
def backend() -> MockupSimBackend:
    """Simulation backend: no audio, no IMU (inherits the None default)."""
    backend = MockupSimBackend(use_audio=False)
    # Seed the kinematics the way `run()` does on entry, so /state/full's
    # default pose fields resolve without spinning the control loop.
    backend.update_head_kinematics_model(
        backend._head_joint_positions,
        backend._antenna_joint_positions,
    )
    return backend


@pytest.fixture
def with_imu(backend, monkeypatch) -> MockupSimBackend:
    """Same backend, but serving a fixed IMU reading (no hardware)."""
    monkeypatch.setattr(backend, "get_imu_data", lambda: READING)
    return backend


# --------------------------------------------------------------------
# Fail-silent: no IMU on this hardware
# --------------------------------------------------------------------


def test_get_imu_without_hardware(router_app, backend) -> None:
    """GET /state/imu -> 200 with a null body when there is no IMU."""
    client = router_app(state.router, backend=backend)

    resp = client.get("/state/imu")

    assert resp.status_code == 200
    assert resp.json() is None


def test_full_state_without_hardware(router_app, backend) -> None:
    """/state/full?with_imu=true -> the key is present and null."""
    client = router_app(state.router, backend=backend)

    resp = client.get("/state/full", params={"with_imu": True})

    assert resp.status_code == 200
    assert resp.json()["imu"] is None


def test_full_state_without_flag(router_app, with_imu) -> None:
    """Without ``with_imu`` the IMU is not read: the field stays null."""
    client = router_app(state.router, backend=with_imu)

    resp = client.get("/state/full")

    assert resp.status_code == 200
    assert resp.json().get("imu") is None


# --------------------------------------------------------------------
# Positive path: a reading is served, without the WS discriminator
# --------------------------------------------------------------------


def test_imu_reading_on_both_routes(router_app, with_imu) -> None:
    """``/state/imu`` and ``/state/full?with_imu=true`` serve the same object.

    Only the four ``ImuData`` fields: the legacy ``type`` discriminator
    (``ImuDataMsg``) never reaches the client on either route.
    """
    client = router_app(state.router, backend=with_imu)

    direct = client.get("/state/imu")
    full = client.get("/state/full", params={"with_imu": True})

    assert direct.status_code == full.status_code == 200
    assert direct.json() == full.json()["imu"] == EXPECTED


# --------------------------------------------------------------------
# State snapshot (WebRTC pose push + get_state reply)
# --------------------------------------------------------------------


def test_snapshot_carries_imu(with_imu) -> None:
    """The snapshot serializes the reading through the ``ImuData`` schema."""
    payload = json.loads(with_imu.build_state_snapshot().model_dump_json())

    assert payload["imu"] == EXPECTED
    assert "type" not in payload["imu"]


def test_snapshot_without_hardware(backend) -> None:
    """No IMU -> the snapshot still has the key, set to null."""
    payload = json.loads(backend.build_state_snapshot().model_dump_json())

    assert payload["imu"] is None
