import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini
import tkinter as tk

def main():
    with ReachyMini() as mini:
        t0 = time.time()
    
        root = tk.Tk()
        root.title("Set Head Euler Angles")

        roll_var = tk.DoubleVar(value=0.0)
        pitch_var = tk.DoubleVar(value=0.0)
        yaw_var = tk.DoubleVar(value=0.0)

        tk.Label(root, text="Roll (deg):").grid(row=0, column=0)
        tk.Scale(root, variable=roll_var, from_=-90, to=90, orient=tk.HORIZONTAL).grid(
            row=0, column=1
        )
        tk.Label(root, text="Pitch (deg):").grid(row=1, column=0)
        tk.Scale(root, variable=pitch_var, from_=-90, to=90, orient=tk.HORIZONTAL).grid(
            row=1, column=1
        )
        tk.Label(root, text="Yaw (deg):").grid(row=2, column=0)
        tk.Scale(root, variable=yaw_var, from_=-90, to=90, orient=tk.HORIZONTAL).grid(
            row=2, column=1
        )

        # Add sliders for X, Y, Z position
        x_var = tk.DoubleVar(value=0.0)
        y_var = tk.DoubleVar(value=0.0)
        z_var = tk.DoubleVar(value=0.0)

        tk.Label(root, text="X (m):").grid(row=3, column=0)
        tk.Scale(
            root,
            variable=x_var,
            from_=-0.05,
            to=0.05,
            resolution=0.001,
            orient=tk.HORIZONTAL,
        ).grid(row=3, column=1)
        tk.Label(root, text="Y (m):").grid(row=4, column=0)
        tk.Scale(
            root,
            variable=y_var,
            from_=-0.05,
            to=0.05,
            resolution=0.001,
            orient=tk.HORIZONTAL,
        ).grid(row=4, column=1)
        tk.Label(root, text="Z (m):").grid(row=5, column=0)
        tk.Scale(
            root,
            variable=z_var,
            from_=-0.05,
            to=0.05,
            resolution=0.001,
            orient=tk.HORIZONTAL,
        ).grid(row=5, column=1)

        # add a checkbox to enable/disable collision checking
        collision_check_var = tk.BooleanVar(value=False)
        tk.Checkbutton(root, text="Check Collision", variable=collision_check_var).grid(
            row=6, column=1
        )

        # Run the GUI in a non-blocking way
        root.update()

        while True:
            t = time.time() - t0
            target = np.deg2rad(30) * np.sin(2 * np.pi * 0.5 * t)

            head = np.eye(4)
            head[:3, 3] = [0, 0, 0.0]

            # Read values from the GUI
            roll = np.deg2rad(roll_var.get())
            pitch = np.deg2rad(pitch_var.get())
            yaw = np.deg2rad(yaw_var.get())
            head[:3, :3] = R.from_euler(
                "xyz", [roll, pitch, yaw], degrees=False
            ).as_matrix()
            head[:3, 3] = [x_var.get(), y_var.get(), z_var.get()]

            root.update()

            mini.set_target(
                head=head,
                antennas=np.array([target, -target]),
                check_collision=collision_check_var.get(),
            )

if __name__ == "__main__":
    main()
