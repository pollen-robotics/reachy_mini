#!/usr/bin/env python3
"""Test script for relative motion timeout and smooth decay functionality.

This script demonstrates the automatic timeout behavior:
1. Applies relative motion for a few seconds
2. Stops sending relative commands
3. Observes automatic smooth decay back to zero over 1 second
4. Total timeout behavior: 1s timeout + 1s decay = 2s to full reset
"""

import math
import time

from reachy_mini import ReachyMini, utils


def main():
    """Test relative motion timeout and decay."""
    print("Testing relative motion timeout and decay...")
    print("1s timeout + 1s smooth decay = 2s total to reset")

    # Motion parameters
    amplitude_deg = 5.0  # degrees
    frequency_hz = 2.0  # Hz
    control_period = 0.02  # 50Hz

    with ReachyMini() as mini:
        print("Connected to robot. Starting test sequence...")
        mini.wake_up()

        print("\n--- Phase 1: Active relative motion (5 seconds) ---")
        start_time = time.time()

        # Phase 1: Send relative commands for 5 seconds
        while time.time() - start_time < 5.0:
            elapsed_time = time.time() - start_time

            # Generate sinusoidal pitch offset
            pitch_offset_rad = math.radians(amplitude_deg) * math.sin(
                2 * math.pi * frequency_hz * elapsed_time
            )

            relative_pose = utils.create_head_pose(
                0,
                0,
                0,  # no translation
                0,
                pitch_offset_rad,
                0,  # only pitch rotation
                degrees=False,
            )

            mini.set_target(head=relative_pose, body_yaw=0.0, is_relative=True)

            print(
                f"\rActive motion: {elapsed_time:.1f}/5.0s, pitch={math.degrees(pitch_offset_rad):.1f}°",
                end="",
            )
            time.sleep(control_period)

        print("\n\n--- Phase 2: Stop relative commands, observe timeout decay ---")
        print("Expecting:")
        print("- 0-1s: Motion continues (no timeout yet)")
        print("- 1-2s: Smooth decay to zero")
        print("- 2s+: Motion should be at zero")

        # Phase 2: Stop sending relative commands, let timeout and decay happen
        stop_time = time.time()
        observation_duration = 4.0  # Observe for 4 seconds total

        while time.time() - stop_time < observation_duration:
            elapsed_since_stop = time.time() - stop_time

            # Don't send any relative commands - just observe
            print(f"\rObserving timeout: {elapsed_since_stop:.1f}/4.0s", end="")
            time.sleep(0.1)  # Slower update rate for observation

        print(
            "\n\n--- Phase 3: Large multi-DOF relative motion (to test interpolation) ---"
        )
        print("Testing X+2cm, Z+1cm, Roll+15° with timeout/decay")
        resume_time = time.time()

        # Phase 3: Large motion with multiple DOFs for 3 seconds
        while time.time() - resume_time < 3.0:
            elapsed_time = time.time() - resume_time

            # Generate multi-DOF pattern with different frequencies for complex motion
            x_offset_m = 0.02 * math.sin(
                2 * math.pi * 1.0 * elapsed_time
            )  # 2cm amplitude, 1Hz
            z_offset_m = 0.01 * math.sin(
                2 * math.pi * 1.3 * elapsed_time
            )  # 1cm amplitude, 1.3Hz
            roll_offset_rad = math.radians(15.0) * math.sin(
                2 * math.pi * 0.7 * elapsed_time
            )  # 15° amplitude, 0.7Hz

            relative_pose = utils.create_head_pose(
                x_offset_m,
                0,
                z_offset_m,  # X and Z translation
                roll_offset_rad,
                0,
                0,  # Roll rotation only
                degrees=False,
            )

            mini.set_target(head=relative_pose, body_yaw=0.0, is_relative=True)

            print(
                f"\rLarge motion: {elapsed_time:.1f}/3.0s | X={x_offset_m * 1000:.1f}mm, Z={z_offset_m * 1000:.1f}mm, Roll={math.degrees(roll_offset_rad):.1f}°",
                end="",
            )
            time.sleep(control_period)

        print("\n\n--- Phase 4: Stop and observe timeout decay of large motion ---")
        print("Should smoothly decay from large offsets back to zero over 2 seconds")

        # Phase 4: Stop sending commands and watch decay of large motion
        stop_time2 = time.time()
        observation_duration2 = 4.0  # Observe for 4 seconds total

        while time.time() - stop_time2 < observation_duration2:
            elapsed_since_stop = time.time() - stop_time2

            # Don't send any relative commands - just observe the decay
            print(
                f"\rObserving large motion decay: {elapsed_since_stop:.1f}/4.0s", end=""
            )
            time.sleep(0.1)  # Slower update rate for observation

        print("\n\n--- Test Complete ---")
        print("Putting robot to sleep...")
        mini.goto_sleep()
        print("Done!")


if __name__ == "__main__":
    main()
