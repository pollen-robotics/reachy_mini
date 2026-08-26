"""Tests for the head-tracking re-aim policy (continuous / periodic / speaking)."""

import time
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

import reachy_mini.daemon.backend.abstract as abstract_module
from reachy_mini.daemon.backend.mockup_sim.backend import MockupSimBackend
from reachy_mini.io.protocol import SetHeadTrackingCmd
from reachy_mini.vision.face_tracking import TrackingTarget


class _Clock:
    """Controllable monotonic clock."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t


class _SpyTracker:
    """Face tracker stand-in with a settable latest target."""

    def __init__(self) -> None:
        self.target: TrackingTarget | None = None
        self.started = False

    def start(self, camera_specs: object) -> None:
        self.started = True

    def set_active(self, active: bool) -> None:
        pass

    def publish_head_pose(self, roll: float, pitch: float, yaw: float) -> None:
        pass

    def latest(self) -> TrackingTarget | None:
        return self.target

    def stop(self) -> None:
        pass


def _target(seq: int, yaw: float) -> TrackingTarget:
    return TrackingTarget(
        seq=seq,
        detected=True,
        roll=0.0,
        pitch=0.0,
        yaw=yaw,
        x=0.0,
        y=0.0,
        face_roll=0.0,
    )


def _make(
    monkeypatch: pytest.MonkeyPatch, mode: str, **fields: object
) -> tuple[MockupSimBackend, _SpyTracker, _Clock]:
    clock = _Clock()
    monkeypatch.setattr(
        abstract_module, "time", SimpleNamespace(monotonic=clock.now, time=time.time)
    )
    backend = MockupSimBackend(use_audio=False)
    backend.current_head_pose = np.eye(4, dtype=np.float64)
    backend._tracking_enabled = True
    backend._tracking_mode = mode  # type: ignore[assignment]
    for name, value in fields.items():
        setattr(backend, name, value)
    tracker = _SpyTracker()
    backend._tracker = tracker  # type: ignore[assignment]
    return backend, tracker, clock


def _publish_and_step(
    backend: MockupSimBackend,
    tracker: _SpyTracker,
    clock: _Clock,
    t: float,
    seq: int,
    yaw: float,
) -> float | None:
    clock.t = t
    tracker.target = _target(seq, yaw)
    backend.step_head_tracking()
    rpy = backend._tracking_target_rpy
    return None if rpy is None else rpy[2]


def test_continuous_mode_adopts_every_fresh_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Continuous mode keeps following the face."""
    backend, tracker, clock = _make(monkeypatch, "continuous")
    assert _publish_and_step(backend, tracker, clock, 0.0, 1, 0.2) == 0.2
    assert _publish_and_step(backend, tracker, clock, 0.1, 2, 0.4) == 0.4


def test_periodic_mode_aims_once_then_waits_for_the_next_glance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Periodic mode adopts the first target, then only one target per interval."""
    backend, tracker, clock = _make(
        monkeypatch, "periodic", _tracking_glance_interval_s=(10.0, 10.0)
    )
    assert _publish_and_step(backend, tracker, clock, 0.0, 1, 0.2) == 0.2  # first aim
    assert _publish_and_step(backend, tracker, clock, 1.0, 2, 0.4) == 0.2  # holding
    assert (
        _publish_and_step(backend, tracker, clock, 9.0, 3, 0.5) == 0.2
    )  # still holding
    assert _publish_and_step(backend, tracker, clock, 10.0, 4, 0.6) == 0.6  # glance
    assert (
        _publish_and_step(backend, tracker, clock, 15.0, 5, 0.8) == 0.6
    )  # holding again
    assert backend._tracking_next_glance_at == 20.0


def test_speaking_mode_follows_only_while_speech_offsets_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Speaking mode holds when silent and follows within the hold window after speech."""
    backend, tracker, clock = _make(
        monkeypatch, "speaking", _tracking_speaking_hold_s=1.0
    )
    assert _publish_and_step(backend, tracker, clock, 0.0, 1, 0.2) == 0.2  # first aim
    assert (
        _publish_and_step(backend, tracker, clock, 1.0, 2, 0.4) == 0.2
    )  # silent: hold

    clock.t = 2.0
    backend.set_speech_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.01))  # wobbler moving
    assert _publish_and_step(backend, tracker, clock, 2.5, 3, 0.6) == 0.6  # speaking
    assert _publish_and_step(backend, tracker, clock, 2.9, 4, 0.7) == 0.7  # within hold
    assert (
        _publish_and_step(backend, tracker, clock, 4.0, 5, 0.9) == 0.7
    )  # hold expired


def test_speaking_mode_uses_media_server_audio_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audio leaving the speaker counts as speaking even without speech offsets."""
    audio = {"ts": None}
    backend, tracker, clock = _make(
        monkeypatch,
        "speaking",
        _tracking_speaking_hold_s=1.0,
        _media_server=SimpleNamespace(last_audio_output_time=lambda: audio["ts"]),
    )
    assert _publish_and_step(backend, tracker, clock, 0.0, 1, 0.2) == 0.2
    assert _publish_and_step(backend, tracker, clock, 5.0, 2, 0.4) == 0.2  # silent
    audio["ts"] = 9.8
    assert (
        _publish_and_step(backend, tracker, clock, 10.0, 3, 0.6) == 0.6
    )  # audio 0.2 s ago
    assert (
        _publish_and_step(backend, tracker, clock, 12.0, 4, 0.8) == 0.6
    )  # audio 2.2 s ago


def test_speaking_mode_uses_the_app_declared_turn_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relayed conversation.turn "speaking" counts as speaking until it changes."""
    backend, tracker, clock = _make(
        monkeypatch, "speaking", _tracking_speaking_hold_s=1.0
    )
    assert _publish_and_step(backend, tracker, clock, 0.0, 1, 0.2) == 0.2
    assert _publish_and_step(backend, tracker, clock, 5.0, 2, 0.4) == 0.2  # silent

    backend.note_app_notification("conversation.turn", {"state": "speaking"})
    assert _publish_and_step(backend, tracker, clock, 10.0, 3, 0.6) == 0.6
    assert (
        _publish_and_step(backend, tracker, clock, 40.0, 4, 0.7) == 0.7
    )  # no hold limit while declared

    clock.t = 41.0
    backend.note_app_notification("conversation.turn", {"state": "ready"})
    assert (
        _publish_and_step(backend, tracker, clock, 41.5, 5, 0.8) == 0.8
    )  # within hold
    assert (
        _publish_and_step(backend, tracker, clock, 43.0, 6, 0.9) == 0.8
    )  # hold expired

    backend.note_app_notification("conversation.turn", {"state": "speaking"})
    backend.note_app_notification("app.disconnected", {})
    assert _publish_and_step(backend, tracker, clock, 50.0, 7, 1.0) == 0.8  # cleared


def test_periodic_mode_still_recenters_when_the_face_is_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Face loss handling is independent of the re-aim policy."""
    backend, tracker, clock = _make(
        monkeypatch, "periodic", _tracking_glance_interval_s=(30.0, 30.0)
    )
    assert _publish_and_step(backend, tracker, clock, 0.0, 1, 0.5) == 0.5
    clock.t = backend._tracking_lost_timeout + 0.5  # no fresh publication since
    backend.step_head_tracking()
    assert backend._tracking_target_rpy == (0.0, 0.0, 0.0)


def test_enable_head_tracking_applies_mode_and_resets_the_glance_schedule() -> None:
    """Re-enabling with a new mode applies tunables and re-aims once right away."""
    backend = MockupSimBackend(use_audio=False)
    backend.current_head_pose = np.eye(4, dtype=np.float64)
    backend._media_server = SimpleNamespace(camera_specs=object())  # type: ignore[assignment]
    backend._tracker = _SpyTracker()  # type: ignore[assignment]
    backend._tracking_next_glance_at = 123.0

    assert backend.enable_head_tracking(
        weight=0.8, mode="periodic", glance_interval_s=(3, 4), speaking_hold_s=2.0
    )

    assert backend._tracking_mode == "periodic"
    assert backend._tracking_glance_interval_s == (3.0, 4.0)
    assert backend._tracking_speaking_hold_s == 2.0
    assert backend._tracking_next_glance_at is None


def test_set_head_tracking_command_forwards_the_mode() -> None:
    """The protocol command carries the mode; the response shape is unchanged."""
    backend = MockupSimBackend(use_audio=False)
    backend.current_head_pose = np.eye(4, dtype=np.float64)
    backend._media_server = SimpleNamespace(camera_specs=object())  # type: ignore[assignment]
    backend._tracker = _SpyTracker()  # type: ignore[assignment]

    responses: list[dict[str, object]] = []
    backend.process_command(
        SetHeadTrackingCmd(enabled=True, mode="speaking", speaking_hold_s=0.5),
        send_response=responses.append,
    )

    assert backend._tracking_mode == "speaking"
    assert backend._tracking_speaking_hold_s == 0.5
    assert responses == [
        {"status": "ok", "command": "set_head_tracking", "enabled": True}
    ]


def test_set_head_tracking_cmd_defaults_and_validation() -> None:
    """Defaults keep today's behavior; nonsensical glance ranges are rejected."""
    cmd = SetHeadTrackingCmd(enabled=True)
    assert cmd.mode == "continuous"
    assert cmd.glance_interval_s == (5.0, 30.0)
    assert cmd.speaking_hold_s == 1.0

    with pytest.raises(ValidationError):
        SetHeadTrackingCmd(enabled=True, glance_interval_s=(10.0, 5.0))
    with pytest.raises(ValidationError):
        SetHeadTrackingCmd(enabled=True, glance_interval_s=(0.0, 5.0))
