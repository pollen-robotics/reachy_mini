"""Shutdown regressions with offline listeners and the headless backend."""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from reachy_mini.apps.manager import AppManager
from reachy_mini.daemon.app.routers import move as router
from test_stop_app_clears_move import _fake_current_app


@pytest.mark.asyncio
async def test_stalled_cancel_listener_releases_guard_and_cleans_app(
    sim_backend: Any,
) -> None:
    release = asyncio.Event()

    async def send_json(message: dict) -> None:
        if message["type"] == "move_cancelled":
            await release.wait()

    router.move_listeners.append(SimpleNamespace(send_json=send_json))
    manager = AppManager(daemon=SimpleNamespace(backend=sim_backend))
    manager.current_app = _fake_current_app()
    # Avoid any idle movement: the robot is already at its sleep pose.
    sim_backend.get_current_head_pose = lambda: sim_backend.SLEEP_HEAD_POSE
    sim_backend.set_motor_control_mode = MagicMock()
    move = SimpleNamespace(
        duration=100, sound_path=None, evaluate=lambda t: (None, None, None)
    )
    uid = router.create_move_task(sim_backend.play_move(move)).uuid
    for _ in range(20):
        if sim_backend.is_move_running:
            break
        await asyncio.sleep(0)
    assert sim_backend.is_move_running
    task = router.move_tasks[uid]
    stopping = asyncio.create_task(manager.stop_current_app())
    try:
        done, _ = await asyncio.wait([stopping], timeout=2.3)
        assert done, "shutdown stuck before its two-second move deadline"
        await stopping
        assert manager.current_app is None
        assert not sim_backend.is_move_running
        assert task.done() and uid not in router.move_tasks
        # Exercise the next actual backend move, including stale stop-flag reset.
        await sim_backend.play_move(SimpleNamespace(duration=0, sound_path=None))
        assert not sim_backend.is_move_running
        assert not sim_backend._stop_move_requested
        assert sim_backend._try_start_move()
        sim_backend._end_move()
    finally:
        release.set()
        await asyncio.gather(stopping, return_exceptions=True)
        router.move_listeners.clear()
        await router.cancel_all_move_tasks()


@pytest.mark.asyncio
async def test_cancel_all_signals_every_task_before_waiting() -> None:
    released, started = asyncio.Event(), asyncio.Event()
    cancelled = [asyncio.Event() for _ in range(3)]
    count = 0

    async def move(index: int) -> None:
        nonlocal count
        count += 1
        if count == 3:
            started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled[index].set()
            await released.wait()

    for i in range(3):
        router.create_move_task(move(i))
    await started.wait()
    stopping = asyncio.create_task(router.cancel_all_move_tasks())
    try:
        for _ in range(10):
            await asyncio.sleep(0)
        assert all(e.is_set() for e in cancelled), "cancellation was serialized"
    finally:
        released.set()
        await stopping


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [ValueError, TimeoutError])
async def test_unexpected_listener_failure_is_observable(
    caplog: pytest.LogCaptureFixture, error_type: type[Exception]
) -> None:
    calls = 0

    async def send_json(message: dict) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error_type("broken notification serializer")

    router.move_listeners.append(SimpleNamespace(send_json=send_json))
    uid = router.create_move_task(asyncio.sleep(0)).uuid
    task = router.move_tasks[uid]
    try:
        await task
        assert "broken notification serializer" in caplog.text
    finally:
        router.move_listeners.clear()


@pytest.mark.asyncio
async def test_cancellation_timeout_retains_ownership_and_blocks_idle_reset() -> None:
    release, started = asyncio.Event(), asyncio.Event()

    async def stubborn_move() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()

    uid = router.create_move_task(stubborn_move()).uuid
    await started.wait()
    backend = SimpleNamespace(is_move_running=True, request_stop_move=MagicMock())
    manager = AppManager(daemon=SimpleNamespace(backend=backend))
    manager.current_app = _fake_current_app()
    stopping = asyncio.create_task(manager.stop_current_app())
    try:
        done, _ = await asyncio.wait([stopping], timeout=2.3)
        assert done, "non-cooperative work escaped the deadline"
        with pytest.raises(TimeoutError):
            await stopping
        assert manager.is_app_running(), "a failed stop must prevent a new app starting"
        assert manager.current_app.status.state == "error"
        assert uid in router.move_tasks, "unfinished move lost its owner"
    finally:
        release.set()
        await asyncio.gather(stopping, return_exceptions=True)
        await router.cancel_all_move_tasks()


@pytest.mark.asyncio
async def test_stop_cancels_unstarted_move_even_when_backend_is_idle() -> None:
    """A task announcing its start cannot outlive the app and begin a stale move."""
    announced, release = asyncio.Event(), asyncio.Event()
    ran = False

    async def send_json(message: dict) -> None:
        if message["type"] == "move_started":
            announced.set()
            await release.wait()

    async def move() -> None:
        nonlocal ran
        ran = True

    backend = SimpleNamespace(
        is_move_running=False,
        get_current_head_pose=lambda: 0,
        SLEEP_HEAD_POSE=0,
    )
    # No physical idle operation; only this existing pose-comparison seam is mocked.
    from unittest.mock import patch

    backend.set_motor_control_mode = MagicMock()
    manager = AppManager(daemon=SimpleNamespace(backend=backend))
    manager.current_app = _fake_current_app()
    router.move_listeners.append(SimpleNamespace(send_json=send_json))
    coro = move()
    uid = router.create_move_task(coro).uuid
    task = router.move_tasks[uid]
    await announced.wait()
    try:
        with patch(
            "reachy_mini.apps.manager.distance_between_poses", return_value=(0, 0, 0)
        ):
            await manager.stop_current_app()
        assert task.done() and coro.cr_frame is None
        release.set()
        await asyncio.sleep(0)
        assert not ran
    finally:
        release.set()
        router.move_listeners.clear()
        await router.cancel_all_move_tasks()
