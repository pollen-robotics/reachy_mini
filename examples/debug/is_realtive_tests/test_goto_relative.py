#!/usr/bin/env python3
"""Test script for goto_target with is_relative functionality.

This script demonstrates:
1. Continuous absolute pitch sine wave (base motion)
2. Relative rectangular motion using goto_target (layered on top)  
3. Timeout behavior when relative motion stops

Rectangle pattern: X constant, Z ±1cm, Y ±2cm
"""

import math
import threading
import time

from reachy_mini import ReachyMini, utils


def absolute_pitch_thread(mini: ReachyMini, stop_event: threading.Event):
    """Continuous absolute pitch sine wave in background."""
    print("Starting absolute pitch motion thread (2Hz, 3°)")
    
    start_time = time.time()
    
    while not stop_event.is_set():
        elapsed_time = time.time() - start_time
        
        # Generate continuous pitch sine wave
        pitch_rad = math.radians(3.0) * math.sin(2 * math.pi * 2.0 * elapsed_time)
        
        absolute_pose = utils.create_head_pose(0, 0, 0, 0, pitch_rad, 0, degrees=False)
        
        mini.set_target(head=absolute_pose, is_relative=False)
        time.sleep(0.02)  # 50Hz
    
    print("Absolute motion thread stopped.")


def main():
    """Test goto_target with relative rectangular motion."""
    print("Testing goto_target with is_relative=True")
    print("Rectangle: X=0, Z=±1cm, Y=±2cm with 2s duration per side")
    
    with ReachyMini() as mini:
        print("Connected to robot. Starting test sequence...")
        mini.wake_up()
        
        # Start absolute pitch motion in background
        stop_event = threading.Event()
        pitch_thread = threading.Thread(
            target=absolute_pitch_thread,
            args=(mini, stop_event),
            daemon=True
        )
        pitch_thread.start()
        
        try:
            print("\n--- Phase 1: Rectangular relative motion (16 seconds total) ---")
            
            # Rectangle corners: (Y, Z) coordinates, X stays at 0
            rectangle_points = [
                (0.02, 0.01),   # Y=+2cm, Z=+1cm
                (0.02, -0.01),  # Y=+2cm, Z=-1cm  
                (-0.02, -0.01), # Y=-2cm, Z=-1cm
                (-0.02, 0.01),  # Y=-2cm, Z=+1cm
            ]
            
            for i, (y, z) in enumerate(rectangle_points):
                corner_name = ["top-right", "bottom-right", "bottom-left", "top-left"][i]
                print(f"Moving to {corner_name}: Y={y*1000:.0f}mm, Z={z*1000:.0f}mm")
                
                relative_pose = utils.create_head_pose(0, y, z, 0, 0, 0, degrees=False)
                
                # Use relative goto_target
                mini.goto_target(
                    head=relative_pose,
                    duration=2.0,
                    method="minjerk",
                    is_relative=True
                )
                time.sleep(0.5)  # Brief pause at each corner
            
            print(f"\n--- Phase 2: Complete rectangle cycle (repeat once more) ---")
            
            # Second rectangle cycle
            for i, (y, z) in enumerate(rectangle_points):
                corner_name = ["top-right", "bottom-right", "bottom-left", "top-left"][i]
                print(f"Moving to {corner_name}: Y={y*1000:.0f}mm, Z={z*1000:.0f}mm")
                
                relative_pose = utils.create_head_pose(0, y, z, 0, 0, 0, degrees=False)
                
                mini.goto_target(
                    head=relative_pose,
                    duration=2.0,
                    method="minjerk", 
                    is_relative=True
                )
                time.sleep(0.5)
            
            print(f"\n--- Phase 3: Stop relative commands, observe timeout decay ---")
            print("Expecting 1s timeout + 1s smooth decay = 2s total")
            print("Rectangle motion should smoothly return to zero while pitch continues")
            
            # Stop sending relative commands and observe timeout
            timeout_start = time.time()
            while time.time() - timeout_start < 5.0:
                elapsed = time.time() - timeout_start
                print(f"\rObserving timeout decay: {elapsed:.1f}/5.0s", end="")
                time.sleep(0.2)
            
            print("\n\n--- Test Complete ---")
            
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