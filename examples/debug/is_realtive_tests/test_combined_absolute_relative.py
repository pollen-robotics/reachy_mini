#!/usr/bin/env python3
"""Test script combining absolute goto_target with relative dance moves.

This script demonstrates the full power of the relative motion system:
1. Absolute rectangular motion using goto_target (base choreography)
2. Relative dance moves using play_on (expressive layer)
3. Both running simultaneously with independent timing
4. Timeout behavior demonstration

This shows how you can have a planned choreography (absolute) with
expressive flourishes (relative) layered on top.
"""

import threading
import time

from reachy_mini import ReachyMini, utils
from reachy_mini.motion.dance.dance_move import DanceMove


def absolute_rectangle_thread(
    mini: ReachyMini, stop_event: threading.Event, cycle_time: float = 12.0
):
    """Continuous absolute rectangular motion in background."""
    print(f"Starting absolute rectangle motion thread (cycle time: {cycle_time}s)")

    # Rectangle corners in world coordinates (X, Y, Z in meters)
    rectangle_points = [
        (0.0, 0.03, 0.015),  # Y=+3cm, Z=+1.5cm
        (0.0, 0.03, -0.015),  # Y=+3cm, Z=-1.5cm
        (0.0, -0.03, -0.015),  # Y=-3cm, Z=-1.5cm
        (0.0, -0.03, 0.015),  # Y=-3cm, Z=+1.5cm
    ]

    corner_names = ["top-right", "bottom-right", "bottom-left", "top-left"]
    side_duration = cycle_time / len(rectangle_points)

    while not stop_event.is_set():
        for i, (x, y, z) in enumerate(rectangle_points):
            if stop_event.is_set():
                break

            corner_name = corner_names[i]
            print(
                f"[ABS] Moving to {corner_name}: Y={y * 1000:.0f}mm, Z={z * 1000:.0f}mm"
            )

            absolute_pose = utils.create_head_pose(x, y, z, 0, 0, 0, degrees=False)

            # Use absolute goto_target for base choreography
            mini.goto_target(
                head=absolute_pose,
                duration=side_duration,
                method="minjerk",
                is_relative=False,
            )

    print("Absolute rectangle motion thread stopped.")


def relative_dance_thread(
    mini: ReachyMini, stop_event: threading.Event, dance_interval: float = 8.0
):
    """Periodic relative dance moves in background."""
    print(f"Starting relative dance motion thread (interval: {dance_interval}s)")

    # Dance moves to use as relative expressions
    expressive_moves = [
        "simple_nod",
        "yeah_nod",
        "head_tilt_roll",
        "side_to_side_sway",
        "uh_huh_tilt",
    ]

    move_index = 0

    while not stop_event.is_set():
        move_name = expressive_moves[move_index % len(expressive_moves)]

        print(f"[REL] Playing '{move_name}' as relative expression...")

        try:
            move = DanceMove(move_name)
            # Play as relative motion (layered on absolute rectangle)
            move.play_on(mini, repeat=1, is_relative=True)
            print(f"[REL] Completed '{move_name}'")
        except Exception as e:
            print(f"[REL] Error playing {move_name}: {e}")

        move_index += 1

    print("Relative dance motion thread stopped.")


def main():
    """Test combined absolute and relative motion."""
    print("Testing combined absolute goto_target + relative play_on")
    print("Absolute: Rectangle motion (12s cycle)")
    print("Relative: Dance moves (8s interval)")
    print("Both run simultaneously and independently!")

    with ReachyMini() as mini:
        print("Connected to robot. Starting combined motion test...")
        mini.wake_up()
        time.sleep(1.0)

        # Create stop event for coordinating threads
        stop_event = threading.Event()

        # Start absolute rectangle motion
        rectangle_thread = threading.Thread(
            target=absolute_rectangle_thread,
            args=(mini, stop_event, 12.0),  # 12s cycle time
            daemon=True,
        )

        # Start relative dance expressions
        dance_thread = threading.Thread(
            target=relative_dance_thread,
            args=(mini, stop_event, 8.0),  # 8s interval
            daemon=True,
        )

        try:
            print("\n--- Phase 1: Combined motion (30 seconds) ---")
            print("Watch how dance expressions layer on top of rectangle motion!")

            rectangle_thread.start()
            dance_thread.start()

            # Let combined motion run
            test_duration = 30.0
            start_time = time.time()

            while time.time() - start_time < test_duration:
                elapsed = time.time() - start_time
                print(
                    f"\rCombined motion running: {elapsed:.1f}/{test_duration}s", end=""
                )
                time.sleep(1.0)

            print("\n\n--- Phase 2: Stop relative motion, continue absolute ---")
            print("Stopping dance expressions, rectangle continues...")

            # Stop dance thread but continue rectangle
            dance_thread.join(timeout=1.0)  # Wait for current move to finish

            # Continue rectangle for 10 more seconds
            continue_duration = 10.0
            start_time = time.time()

            while time.time() - start_time < continue_duration:
                elapsed = time.time() - start_time
                print(
                    f"\rOnly absolute motion: {elapsed:.1f}/{continue_duration}s",
                    end="",
                )
                time.sleep(1.0)

            print("\n\n--- Phase 3: Timeout behavior test ---")
            print("Stopping all motion, observing relative offset decay...")

            # Stop all motion and observe timeout
            stop_event.set()
            rectangle_thread.join(timeout=2.0)

            timeout_start = time.time()
            while time.time() - timeout_start < 4.0:
                elapsed = time.time() - timeout_start
                print(f"\rObserving timeout decay: {elapsed:.1f}/4.0s", end="")
                time.sleep(0.5)

            print("\n\n--- Test Complete ---")

        except KeyboardInterrupt:
            print("\nTest interrupted by user.")
        finally:
            stop_event.set()
            print("Stopping all motion threads...")
            rectangle_thread.join(timeout=2.0)
            dance_thread.join(timeout=2.0)
            time.sleep(1.0)
            print("Putting robot to sleep...")
            mini.goto_sleep()
            print("Done!")


if __name__ == "__main__":
    main()
