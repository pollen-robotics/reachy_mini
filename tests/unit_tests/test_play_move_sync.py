"""The blocking play_move runs async_play_move in the caller's thread."""

import threading

from reachy_mini import ReachyMini


def test_play_move_runs_in_the_caller_thread() -> None:
    """Arguments pass through, and Ctrl+C can reach the move (same thread)."""
    calls: list[tuple[object, ...]] = []

    async def fake_async_play_move(*args: object) -> None:
        calls.append((*args, threading.get_ident()))

    robot = ReachyMini.__new__(ReachyMini)
    robot.async_play_move = fake_async_play_move  # type: ignore[method-assign]
    move = object()

    robot.play_move(move, 50.0, 1.0, False)  # type: ignore[arg-type]

    assert calls == [(move, 50.0, 1.0, False, threading.get_ident())]
