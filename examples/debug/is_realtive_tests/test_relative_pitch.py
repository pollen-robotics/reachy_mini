#!/usr/bin/env python3
"""Minimal test script for relative pitch motion.

This script simply applies a continuous sinusoidal pitch offset using the
is_relative=True feature to verify basic relative movement functionality.
"""

import math
import time

from reachy_mini import ReachyMini, utils


def main():
    """Test pure relative pitch motion."""
    print("Testing relative pitch motion...")
    
    # Motion parameters
    amplitude_deg = 5.0  # degrees
    frequency_hz = 2.0   # Hz
    control_period = 0.02  # 50Hz
    
    print(f"Amplitude: {amplitude_deg}° | Frequency: {frequency_hz}Hz")
    print("Press Ctrl+C to stop")
    
    with ReachyMini() as mini:
        print("Connected to robot. Starting relative pitch motion...")
        mini.wake_up()
        
        start_time = time.time()
        
        try:
            while True:
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # Generate sinusoidal pitch offset
                pitch_offset_rad = math.radians(amplitude_deg) * \
                                  math.sin(2 * math.pi * frequency_hz * elapsed_time)
                
                # Create relative pose with only pitch rotation
                relative_pose = utils.create_head_pose(
                    0, 0, 0,  # no translation
                    0, pitch_offset_rad, 0,  # only pitch rotation
                    degrees=False
                )
                
                # Send relative command
                mini.set_target(
                    head=relative_pose,
                    body_yaw=0.0,
                    is_relative=True  # This is the key test
                )
                
                time.sleep(control_period)
                
        except KeyboardInterrupt:
            print("\nCtrl-C received. Stopping...")
        finally:
            print("Putting robot to sleep...")
            mini.goto_sleep()
            print("Done.")


if __name__ == "__main__":
    main()