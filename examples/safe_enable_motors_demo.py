"""Demo: re-enabling torque after manual displacement must NOT snap.

Reproduces the bug the recent enable_motors() fix addresses, end-to-end:

  1. Go to base pose.
  2. Disable torque. The head falls under gravity.
  3. Manually move the head somewhere different (you have 5 s).
  4. Re-enable torque. <-- this used to snap the head back to base.
  5. Wait 1 s.
  6. Go to base pose smoothly.

With the fix, step 4 leaves the head right where you left it.
Without the fix, step 4 snaps it back toward the previous goal.
"""

import time

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini(media_backend="no_media") as mini:
    print("Going to base pose…")
    mini.goto_target(create_head_pose(), antennas=[0.0, 0.0], duration=1.0)
    time.sleep(1.2)

    print("Disabling torque — move the head by hand (2 s)…")
    mini.disable_motors()
    time.sleep(2.0)

    print("Re-enabling torque (must not move!)…")
    mini.enable_motors()
    time.sleep(1.0)

    print("Going to base pose…")
    mini.goto_target(create_head_pose(), antennas=[0.0, 0.0], duration=1.0)
    time.sleep(1.2)

    print("Done.")
