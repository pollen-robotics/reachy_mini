"""Enable daemon-side head tracking and print the tracked face target.

Reachy Mini turns its head to follow the closest face. Detection runs inside
the daemon, so this script only toggles tracking and reads the result.

Note:
    The daemon must be running before executing this script.

"""

# START doc_example

from reachy_mini import ReachyMini

with ReachyMini() as mini:
    mini.start_head_tracking()
    try:
        while True:
            # Waits for the next daemon status broadcast, so the loop runs at ~1 Hz.
            face = mini.get_tracked_face()
            if face.detected:
                print(f"Face at x={face.x:+.2f}, y={face.y:+.2f}")
            else:
                print("No face detected")
    except KeyboardInterrupt:
        pass
    finally:
        mini.stop_head_tracking()

# END doc_example
