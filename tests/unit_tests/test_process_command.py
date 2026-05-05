"""Tests for Backend.process_command, the transport-agnostic command dispatcher.

These tests guard the regression fixed in commit 06107016: when a
``GotoTargetCmd`` arrives via WebRTC (or any JSON transport), its ``head``
field is a flattened 16-element list. The backend must reshape it to ``(4, 4)``
before passing to ``goto_target``, otherwise downstream code (``GotoMove``)
receives a 1D array and the move silently breaks.

The bug was not caught earlier because the Python SDK's ``set_target_head_pose``
also flattens, but ``goto_target`` was historically called as a Python method
on a 4x4 ``np.ndarray`` and never went through the JSON command path.
"""

import asyncio
import json
from typing import Any, Callable

import numpy as np
import pytest

from reachy_mini.daemon.backend.mockup_sim.backend import MockupSimBackend
from reachy_mini.io.protocol import GotoTargetCmd


def _make_backend() -> MockupSimBackend:
    """Build a lightweight backend with no audio, no kinematics warmup."""
    return MockupSimBackend(use_audio=False)


def _make_pose_matrix() -> np.ndarray:
    """A non-identity 4x4 pose so we can detect any axis-mixing bug."""
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [0.1, 0.2, 0.3]
    pose[0, 0] = 0.5
    pose[1, 1] = 0.6
    return pose


def _patch_async_goto(
    backend: MockupSimBackend,
) -> dict[str, Any]:
    """Replace ``_async_goto`` with a spy that records call arguments.

    Returns the dict the spy writes into.
    """
    captured: dict[str, Any] = {}

    async def fake_async_goto(
        send_response: Callable[[dict[str, Any]], None],
        head: Any,
        antennas: Any,
        duration: float,
        body_yaw: float | None,
    ) -> None:
        captured["head"] = head
        captured["antennas"] = antennas
        captured["duration"] = duration
        captured["body_yaw"] = body_yaw
        send_response({"status": "ok", "command": "goto_target", "completed": True})

    backend._async_goto = fake_async_goto  # type: ignore[assignment]
    return captured


@pytest.mark.asyncio
async def test_goto_target_reshapes_flat_head_to_4x4() -> None:
    """A flat 16-element head from JSON must reach goto_target as a 4x4 array.

    Before the fix, ``np.array(cmd.head)`` produced a shape-(16,) array, which
    silently broke goto via WebRTC.
    """
    backend = _make_backend()
    captured = _patch_async_goto(backend)

    pose = _make_pose_matrix()
    flat_head = pose.flatten().tolist()
    assert len(flat_head) == 16

    responses: list[dict[str, Any]] = []
    cmd = GotoTargetCmd(head=flat_head, duration=0.25, body_yaw=0.1)
    backend.process_command(cmd, send_response=responses.append)

    # process_command schedules _async_goto via asyncio.create_task; yield once
    # so the scheduled coroutine runs and the spy records its arguments.
    await asyncio.sleep(0)

    head = captured["head"]
    assert isinstance(head, np.ndarray)
    assert head.shape == (4, 4), (
        f"GotoTargetCmd head should be reshaped to (4, 4), got {head.shape}"
    )
    np.testing.assert_array_equal(head, pose)
    assert captured["duration"] == 0.25
    assert captured["body_yaw"] == 0.1
    assert captured["antennas"] is None


@pytest.mark.asyncio
async def test_goto_target_passes_none_when_head_omitted() -> None:
    """Antennas-only goto must keep head as None (no spurious reshape)."""
    backend = _make_backend()
    captured = _patch_async_goto(backend)

    cmd = GotoTargetCmd(antennas=[0.1, -0.1], duration=0.5)
    backend.process_command(cmd, send_response=lambda _: None)
    await asyncio.sleep(0)

    assert captured["head"] is None
    np.testing.assert_array_equal(captured["antennas"], np.array([0.1, -0.1]))


@pytest.mark.asyncio
async def test_goto_target_via_webrtc_json_message() -> None:
    """Full WebRTC entry path: JSON string in, 4x4 head out.

    This reproduces the exact code path that triggered the original bug:
    a JSON string arrives on the data channel, gets parsed by
    ``command_adapter``, and is dispatched through ``process_command``.
    """
    backend = _make_backend()
    captured = _patch_async_goto(backend)

    pose = _make_pose_matrix()
    message = json.dumps(
        {
            "type": "goto_target",
            "head": pose.flatten().tolist(),
            "duration": 0.5,
        }
    )

    backend._handle_webrtc_message(peer_id="test-peer", message=message)
    await asyncio.sleep(0)

    head = captured["head"]
    assert isinstance(head, np.ndarray)
    assert head.shape == (4, 4)
    np.testing.assert_array_equal(head, pose)
