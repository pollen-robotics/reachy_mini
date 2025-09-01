#!/usr/bin/env python3
"""Test script for play_on with is_relative functionality.

This script demonstrates:
1. Continuous absolute pitch sine wave (base motion)
2. Relative dance moves using play_on (layered on top)
3. Timeout behavior when relative motion stops

Tests multiple dance moves as relative offsets.
"""

import math
import threading
import time

from reachy_mini import ReachyMini, utils
from reachy_mini.motion.dance_move import DanceMove


def absolute_pitch_thread(mini: ReachyMini, stop_event: threading.Event):
    """Continuous absolute pitch sine wave in background."""
    print("Starting absolute pitch motion thread (1.5Hz, 4°)")

    start_time = time.time()

    while not stop_event.is_set():
        elapsed_time = time.time() - start_time

        # Generate continuous pitch sine wave
        pitch_rad = math.radians(4.0) * math.sin(2 * math.pi * 1.5 * elapsed_time)

        absolute_pose = utils.create_head_pose(0, 0, 0, 0, pitch_rad, 0, degrees=False)

        mini.set_target(head=absolute_pose, is_relative=False)
        time.sleep(0.02)  # 50Hz

    print("Absolute motion thread stopped.")


def main():
    """Test play_on with relative dance moves."""
    print("Testing play_on with is_relative=True")
    print("Playing dance moves as relative offsets on top of absolute pitch motion")

    # Select some expressive moves for relative testing
    test_moves = [
        "simple_nod",
        "side_to_side_sway",
        "head_tilt_roll",
        "yeah_nod",
    ]

    with ReachyMini() as mini:
        print("Connected to robot. Starting test sequence...")
        mini.wake_up()

        # Start absolute pitch motion in background
        stop_event = threading.Event()
        pitch_thread = threading.Thread(
            target=absolute_pitch_thread, args=(mini, stop_event), daemon=True
        )
        pitch_thread.start()

        try:
            print(
                f"\n--- Phase 1: Playing {len(test_moves)} dance moves as relative offsets ---"
            )

            for i, move_name in enumerate(test_moves):
                print(
                    f"\n[{i + 1}/{len(test_moves)}] Playing '{move_name}' as relative motion..."
                )

                # Create dance move and play as relative
                move = DanceMove(move_name)
                print(f"Description: {move.move_metadata['description']}")
                print(f"Duration: {move.duration:.1f}s")

                # Play the move as relative motion (layered on pitch)
                move.play_on(mini, repeat=1, is_relative=True)

                print(f"Completed '{move_name}'")
                time.sleep(1.0)  # Brief pause between moves

            print("\n--- Phase 2: Repeat one move multiple times ---")
            print("Playing 'side_to_side_sway' 3 times as relative motion")

            sway_move = DanceMove("side_to_side_sway")
            sway_move.play_on(mini, repeat=3, is_relative=True)

            print("\n--- Phase 3: Stop relative commands, observe timeout decay ---")
            print("Expecting 1s timeout + 1s smooth decay = 2s total")
            print("Dance motion should smoothly decay while pitch continues")

            # Stop sending relative commands and observe timeout
            timeout_start = time.time()
            while time.time() - timeout_start < 5.0:
                elapsed = time.time() - timeout_start
                print(f"\rObserving timeout decay: {elapsed:.1f}/5.0s", end="")
                time.sleep(0.2)

            print("\n\n--- Phase 4: Resume relative motion to verify recovery ---")
            print("Playing 'simple_nod' to verify system still works after timeout")

            recovery_move = DanceMove("simple_nod")
            recovery_move.play_on(mini, repeat=2, is_relative=True)

            print("\n--- Test Complete ---")

        except KeyboardInterrupt:
            print("\nTest interrupted by user.")
        finally:
            stop_event.set()
            print("Stopping absolute motion...")
            time.sleep(0.5)
            print("Putting robot to sleep...")
            mini.goto_sleep()
            print("Done!")


if __name__ == "__main__":
    main()
