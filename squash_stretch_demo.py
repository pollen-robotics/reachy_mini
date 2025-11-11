"""
Demo: Squash and Stretch Effect for Reachy Mini
Shows how to add cartoon-style deformation to Reachy using simple scaling tricks.
"""

import mujoco
import numpy as np
import time

class SquashStretch:
    """Add squash and stretch effects to a MuJoCo body."""

    def __init__(self, model, data, body_name="base_link"):
        self.model = model
        self.data = data
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.original_sizes = {}

        # Store original geom sizes
        for i in range(model.ngeom):
            if model.geom_bodyid[i] == self.body_id:
                self.original_sizes[i] = model.geom_size[i].copy()

    def bounce(self, time_val, frequency=5.0, amplitude=0.1):
        """Apply bouncing effect (squash/stretch vertically)."""
        # Sine wave for bounce
        bounce_factor = 1.0 + amplitude * abs(np.sin(time_val * frequency * np.pi))

        # Inverse factor to preserve volume (cartoon physics)
        inv_factor = 1.0 / np.sqrt(bounce_factor)

        for geom_id, original_size in self.original_sizes.items():
            # Stretch vertically, squash horizontally
            new_size = original_size.copy()
            if len(new_size) == 3:  # Capsule, cylinder, box
                new_size[0] *= inv_factor  # X
                new_size[1] *= inv_factor  # Y
                new_size[2] *= bounce_factor  # Z (height)
            elif len(new_size) == 1:  # Sphere
                # For sphere, just scale uniformly
                new_size[0] *= bounce_factor

            self.model.geom_size[geom_id] = new_size

    def wiggle(self, time_val, frequency=3.0, amplitude=0.05):
        """Apply wiggle effect (side-to-side scaling variation)."""
        wiggle_x = 1.0 + amplitude * np.sin(time_val * frequency * np.pi)
        wiggle_y = 1.0 + amplitude * np.cos(time_val * frequency * np.pi)

        for geom_id, original_size in self.original_sizes.items():
            new_size = original_size.copy()
            if len(new_size) == 3:
                new_size[0] *= wiggle_x
                new_size[1] *= wiggle_y

            self.model.geom_size[geom_id] = new_size

    def pulse(self, time_val, frequency=2.0, amplitude=0.08):
        """Pulsing effect (uniform scaling)."""
        pulse_factor = 1.0 + amplitude * (0.5 + 0.5 * np.sin(time_val * frequency * 2 * np.pi))

        for geom_id, original_size in self.original_sizes.items():
            self.model.geom_size[geom_id] = original_size * pulse_factor

    def reset(self):
        """Reset to original sizes."""
        for geom_id, original_size in self.original_sizes.items():
            self.model.geom_size[geom_id] = original_size.copy()

    def emotional_deform(self, emotion, intensity=1.0):
        """Apply deformation based on emotional state."""
        t = time.time()

        if emotion == "excited":
            self.bounce(t, frequency=6.0, amplitude=0.12 * intensity)
        elif emotion == "nervous":
            self.wiggle(t, frequency=8.0, amplitude=0.08 * intensity)
        elif emotion == "calm":
            self.pulse(t, frequency=1.0, amplitude=0.03 * intensity)
        elif emotion == "scared":
            # Quick tremble
            self.wiggle(t, frequency=15.0, amplitude=0.05 * intensity)
        elif emotion == "confident":
            # Slow, large pulsing
            self.pulse(t, frequency=0.8, amplitude=0.10 * intensity)
        else:
            self.reset()


# Example usage in desktop viewer:
"""
# In main loop, after loading model:
squasher = SquashStretch(model, data, "base_link")

# In render loop:
sim_time = sim_step * model.opt.timestep

# Apply effect based on current move or emotion
if current_emotion == "excited":
    squasher.emotional_deform("excited", intensity=0.7)
elif just_landed_from_jump:
    squasher.bounce(sim_time, frequency=10.0, amplitude=0.2)
else:
    squasher.reset()
"""

# Performance note:
# This modifies model.geom_size which is FAST (just updating floats)
# NOT modifying mesh vertices (which would require full mesh rebuild)
# Should add < 1ms per frame on modern hardware
